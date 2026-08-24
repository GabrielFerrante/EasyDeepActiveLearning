import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from augmentation import sharpen, mixup

# MixMatch (Berthelot, Carlini, Goodfellow, Papernot, Oliver & Raffel, 2019, NeurIPS, "MixMatch: A
# Holistic Approach to Semi-Supervised Learning") e ReMixMatch (Berthelot, Carlini, Cubuk, Kurakin,
# Sohn, Zhang & Raffel, 2019, ICLR 2020, "ReMixMatch: Semi-Supervised Learning with Distribution
# Alignment and Augmentation Anchoring") — técnicas de TREINO (loop próprio, substituem
# Training/train.py), assumem classificação de rótulo único (assinam p/q como distribuições sobre
# num_classes).
#
# unlabeled_loader precisa ser construído sobre Labeling_Strategies.augmentation.MultiAugmentDataset
# (batch[0] é uma lista de K tensores [Batch,C,H,W], K augmentations independentes da mesma imagem).


def _guess_label(model, augmented_batch, temperature):
    """Label guessing (Eqs. 6-7 do MixMatch): média das predições sobre as K augmentations, depois
    sharpening — usado como pseudo-rótulo "soft" pra cada amostra não rotulada."""
    with torch.no_grad():
        probs_sum = None
        for u_k in augmented_batch:
            probs_k = torch.softmax(model(u_k), dim=-1)
            probs_sum = probs_k if probs_sum is None else probs_sum + probs_k
        q_bar = probs_sum / len(augmented_batch)
    return sharpen(q_bar, temperature)


def _mixmatch_losses(model, x_hat, p_b, u_augs, q_b, alpha, unlabeled_loss="mse"):
    """Monta X̂/Û (Eqs. 10-13), mistura via MixUp e calcula L_X (CE) e L_U — núcleo comum a MixMatch
    e ReMixMatch (Eqs. 2-5 do MixMatch).

    unlabeled_loss controla a forma de L_U: "mse" (Eq. 4 do MixMatch — usado por
    train_model_with_mixmatch) ou "ce" (cross-entropy H(q,p_model), usado por
    train_model_with_remixmatch — o próprio paper do ReMixMatch substitui o MSE do MixMatch por
    cross-entropy nesse termo, Seção 3.2.1: "enabled us to replace MixMatch's unlabeled-data mean
    squared error loss with a standard cross-entropy loss"; o ablation do paper mostra que usar MSE
    aqui piora o erro de 5.94% pra 17.28%)."""
    batch_size = x_hat.size(0)

    all_inputs = torch.cat([x_hat] + u_augs, dim=0)
    all_targets = torch.cat([p_b] + [q_b] * len(u_augs), dim=0)

    perm = torch.randperm(all_inputs.size(0), device=all_inputs.device)
    mixed_inputs, mixed_targets = mixup(all_inputs, all_inputs[perm], all_targets, all_targets[perm], alpha)

    mixed_x, mixed_u = mixed_inputs[:batch_size], mixed_inputs[batch_size:]
    mixed_p, mixed_q = mixed_targets[:batch_size], mixed_targets[batch_size:]

    log_probs_x = torch.log_softmax(model(mixed_x), dim=-1)
    l_x = -(mixed_p * log_probs_x).sum(dim=-1).mean()  # Eq. 3: H(p, p_model)

    if unlabeled_loss == "ce":
        log_probs_u = torch.log_softmax(model(mixed_u), dim=-1)
        l_u = -(mixed_q * log_probs_u).sum(dim=-1).mean()  # ReMixMatch: H(q, p_model)
    else:
        probs_u = torch.softmax(model(mixed_u), dim=-1)
        l_u = ((mixed_q - probs_u) ** 2).mean()  # Eq. 4 do MixMatch: MSE

    return l_x, l_u


def _lambda_u_rampup(lambda_u, step, rampup_steps):
    if not rampup_steps:
        return lambda_u
    return lambda_u * min(1.0, step / rampup_steps)


def train_model_with_mixmatch(
    model, labeled_loader, unlabeled_loader, test_loader, task,
    optimizer, device, epochs, num_classes,
    sharpening_temperature=0.5, alpha=0.75, lambda_u=75.0, lambda_u_rampup_steps=None,
    writer=None, log_prefix="",
):
    """
    Loop de treino do MixMatch (Algorithm 1 do paper). Cada passo:
    1. Label guessing: q_b = Sharpen(média das predições sobre as K augmentations de u_b, T).
    2. Concatena (x̂_b, p_b) rotulado com (û_{b,k}, q_b) não rotulado, embaralha, e aplica MixUp
       (Eqs. 8-11) entre cada item original e um par aleatório do conjunto combinado.
    3. L_X = cross-entropy no lote misto rotulado; L_U = MSE no lote misto não rotulado (Eqs. 3-4).
    4. L = L_X + λ_u·L_U (Eq. 5) — λ_u tipicamente MUITO maior que 1 no MixMatch (paper usa até 75),
       já que L_U é MSE (escala bem menor que cross-entropy).

    labeled_loader: rótulos como índice de classe (convertidos pra one-hot internamente).
    lambda_u_rampup_steps: se informado, λ_u sobe linearmente de 0 até lambda_u ao longo desses passos.

    Requer uma task de classificação de rótulo único (ex.: ClassificationTask) — usada aqui só pra
    avaliação (task.compute_metric), já que L_X/L_U são calculadas internamente (CE + MSE, Eqs. 3-4),
    mas uma task sem atributo .criterion (ex.: DetectionTask, VLMContrastiveTask) indica um formato de
    batch/saída incompatível com a suposição de rótulo único one-hot deste método.
    """
    if not hasattr(task, "criterion"):
        raise AttributeError("MixMatch exige uma task de classificação (ex.: ClassificationTask).")

    total_step = 0
    loss_val = 0.0
    test_metric = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_steps = 0

        unlabeled_iter = iter(unlabeled_loader)
        progress_bar = tqdm(labeled_loader, desc=f"MixMatch [{epoch + 1}/{epochs}]", unit="batch", leave=False)
        for labeled_batch in progress_bar:
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)

            x_hat = labeled_batch[0].to(device)
            y_l = labeled_batch[1].to(device)
            p_b = F.one_hot(y_l, num_classes=num_classes).float()

            u_augs = [u.to(device) for u in unlabeled_batch[0]]
            q_b = _guess_label(model, u_augs, sharpening_temperature)

            optimizer.zero_grad()
            l_x, l_u = _mixmatch_losses(model, x_hat, p_b, u_augs, q_b, alpha)

            current_lambda_u = _lambda_u_rampup(lambda_u, total_step, lambda_u_rampup_steps)
            total_loss = l_x + current_lambda_u * l_u
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            n_steps += 1
            total_step += 1
            progress_bar.set_postfix(l_x=f"{l_x.item():.3f}", l_u=f"{l_u.item():.3f}", lam=f"{current_lambda_u:.1f}")

        loss_val = running_loss / n_steps
        test_metric = _evaluate(model, test_loader, task, device, epoch, epochs)
        print(f"Epoch {epoch + 1}/{epochs}: Loss {loss_val:.4f} | Test Metric: {test_metric:.4f}")

        if writer is not None:
            writer.add_scalar(f"{log_prefix}Loss/train", loss_val, epoch)
            writer.add_scalar(f"{log_prefix}Metric/test", test_metric, epoch)

    return loss_val, test_metric


def train_model_with_remixmatch(
    model, labeled_loader, unlabeled_loader, test_loader, task,
    optimizer, device, epochs, num_classes,
    sharpening_temperature=0.5, alpha=0.75, lambda_u=1.5, lambda_u1=0.5,
    lambda_u_rampup_steps=None, align_momentum=0.999,
    writer=None, log_prefix="",
):
    """
    ReMixMatch = MixMatch + Distribution Alignment + Augmentation Anchoring (Berthelot et al.,
    2019b). Implementa o núcleo dos dois acréscimos do paper em cima de _mixmatch_losses; a loss
    auxiliar de rotation prediction (self-supervised) do paper fica FORA do escopo aqui, por ser uma
    técnica de representação ortogonal a pseudo-labeling (mesmo raciocínio usado pra excluir a loss
    de metric learning do Dynamic Curriculum Learning em Balance_Strategies).

    - Augmentation Anchoring: a label guessing usa uma única augmentation FRACA como "âncora" (em vez
      de média sobre K augmentations independentes do MixMatch) — mais estável, já que a âncora não
      é forte o bastante pra distorcer a predição usada como alvo. unlabeled_loader ainda deve prover
      K versões por amostra (via MultiAugmentDataset); a primeira é tratada como a âncora fraca, as
      demais como augmentations fortes usadas no MixUp.

    - Distribution Alignment: a distribuição guessada q_b é re-escalada pela razão entre a
      distribuição de classes do conjunto ROTULADO (p̃(y), fixa) e a média móvel da distribuição de
      classes prevista pelo modelo no não rotulado (p̃_model(y), atualizada com EMA de fator
      align_momentum) — corrige o viés do modelo em favor das classes que ele já prevê mais,
      encorajando a distribuição marginal das predições no não rotulado a bater com a do rotulado.
      Nota: o paper define p̃_model(y) como a média móvel das predições sobre os ÚLTIMOS 128 BATCHES
      (janela deslizante); aqui é aproximada por uma EMA de decaimento fixo (align_momentum=0.999,
      janela efetiva ~1000 batches) — mais simples de manter em memória, mas não uma leitura literal
      do paper.

    Loss total (Eqs. 3-4, "Putting it all together"): L = L_X + λ_u·L_U + λ_u1·L_Û1.
    - L_X: cross-entropy no lote misto (via MixUp) rotulado.
    - L_U: cross-entropy H(q_b, p_model) no lote misto (via MixUp) não rotulado — DIFERENTE do
      MixMatch, que usa MSE aqui; o paper mostra que a Augmentation Anchoring permite (e prefere)
      cross-entropy nesse termo (ver nota em _mixmatch_losses).
    - L_Û1: termo "pre-mixup" — cross-entropy do MESMO alvo q_b contra a predição do modelo numa
      única augmentation FORTE, SEM passar pelo MixUp (λ_u1=0.5 no paper).

    Requer uma task de classificação de rótulo único (ex.: ClassificationTask) — ver nota equivalente
    em train_model_with_mixmatch.
    """
    if not hasattr(task, "criterion"):
        raise AttributeError("ReMixMatch exige uma task de classificação (ex.: ClassificationTask).")

    labeled_class_distribution = torch.ones(num_classes, device=device) / num_classes
    model_prediction_ema = torch.ones(num_classes, device=device) / num_classes
    distribution_initialized = False

    total_step = 0
    loss_val = 0.0
    test_metric = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_steps = 0

        unlabeled_iter = iter(unlabeled_loader)
        progress_bar = tqdm(labeled_loader, desc=f"ReMixMatch [{epoch + 1}/{epochs}]", unit="batch", leave=False)
        for labeled_batch in progress_bar:
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)

            x_hat = labeled_batch[0].to(device)
            y_l = labeled_batch[1].to(device)
            p_b = F.one_hot(y_l, num_classes=num_classes).float()

            # Distribuição real do rótulo (EMA simples sobre os batches rotulados vistos)
            batch_label_dist = p_b.mean(dim=0)
            labeled_class_distribution = (
                align_momentum * labeled_class_distribution + (1 - align_momentum) * batch_label_dist
                if distribution_initialized else batch_label_dist
            )

            u_augs = [u.to(device) for u in unlabeled_batch[0]]
            weak_anchor, strong_augs = u_augs[0], u_augs[1:] or u_augs[:1]

            with torch.no_grad():
                q_anchor = torch.softmax(model(weak_anchor), dim=-1)  # augmentation anchoring: 1 predição só

            model_prediction_ema = (
                align_momentum * model_prediction_ema + (1 - align_momentum) * q_anchor.mean(dim=0)
                if distribution_initialized else q_anchor.mean(dim=0)
            )
            distribution_initialized = True

            # Distribution alignment (razão rotulado/modelo, renormalizada)
            aligned = q_anchor * (labeled_class_distribution + 1e-8) / (model_prediction_ema + 1e-8)
            aligned = aligned / aligned.sum(dim=-1, keepdim=True)
            q_b = sharpen(aligned, sharpening_temperature)

            optimizer.zero_grad()
            l_x, l_u = _mixmatch_losses(model, x_hat, p_b, strong_augs, q_b, alpha, unlabeled_loss="ce")

            # Termo L_Û1 (pre-mixup): cross-entropy do mesmo alvo q_b contra uma augmentation forte
            # ÚNICA, sem passar pelo MixUp — Eqs. 3-4 do paper.
            pre_mixup_log_probs = torch.log_softmax(model(strong_augs[0]), dim=-1)
            l_u1 = -(q_b * pre_mixup_log_probs).sum(dim=-1).mean()

            current_lambda_u = _lambda_u_rampup(lambda_u, total_step, lambda_u_rampup_steps)
            total_loss = l_x + current_lambda_u * l_u + lambda_u1 * l_u1
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            n_steps += 1
            total_step += 1
            progress_bar.set_postfix(l_x=f"{l_x.item():.3f}", l_u=f"{l_u.item():.3f}",
                                      l_u1=f"{l_u1.item():.3f}", lam=f"{current_lambda_u:.1f}")

        loss_val = running_loss / n_steps
        test_metric = _evaluate(model, test_loader, task, device, epoch, epochs)
        print(f"Epoch {epoch + 1}/{epochs}: Loss {loss_val:.4f} | Test Metric: {test_metric:.4f}")

        if writer is not None:
            writer.add_scalar(f"{log_prefix}Loss/train", loss_val, epoch)
            writer.add_scalar(f"{log_prefix}Metric/test", test_metric, epoch)

    return loss_val, test_metric


def _evaluate(model, loader, task, device, epoch, epochs):
    model.eval()
    all_metrics = []
    progress_bar = tqdm(loader, desc=f"Eval [{epoch + 1}/{epochs}]", unit="batch", leave=False)
    for batch in progress_bar:
        all_metrics.extend(task.compute_metric(model, batch, device))
    return float(np.mean(all_metrics))
