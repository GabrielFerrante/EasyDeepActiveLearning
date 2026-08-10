import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# UDA — Unsupervised Data Augmentation for Consistency Training (Xie, Dai, Hovy, Luong & Le, 2019/2020,
# NeurIPS, "Unsupervised Data Augmentation for Consistency Training"): muito parecido com FixMatch
# (mesmo padrão weak/strong), mas usa uma loss de CONSISTÊNCIA (KL/cross-entropy contra uma
# distribuição-alvo SUAVIZADA — soft label) em vez de comparar contra um pseudo-rótulo hard (argmax).
# Como FixMatch, é uma técnica de TREINO — tem loop próprio, substitui Training/train.py.
#
# unlabeled_loader precisa ser construído sobre Labeling_Strategies.augmentation.WeakStrongAugmentDataset.


def _reduce_sample_dims(tensor):
    while tensor.dim() > 1:
        tensor = tensor.mean(dim=-1)
    return tensor


def sharpen_with_temperature(logits, temperature):
    """p_θ̃^(sharp)(y|x) = exp(z_y/τ) / Σ_y' exp(z_y'/τ) — softmax com temperatura, aplicado à
    distribuição-alvo antes da consistency loss (paper usa τ=0.4 pra CIFAR-10/SVHN/ImageNet)."""
    return torch.softmax(logits / temperature, dim=-1)


def tsa_threshold(step, total_steps, num_classes, schedule="linear", scale=5.0):
    """
    Training Signal Annealing (TSA): threshold crescente η_t que MASCARA exemplos ROTULADOS cuja
    confiança na classe certa já ultrapassa η_t — evita que o treino supervisionado (poucos exemplos)
    convirja rápido demais e passe a "overfitar" antes do sinal não supervisionado (muitos exemplos)
    ter efeito. η_t começa em 1/num_classes (nível de chance) e sobe até 1 conforme o schedule:

        η_t = α_t·(1 - 1/K) + 1/K,   K = num_classes

    schedule controla como α_t sobe de 0 a 1 ao longo do treino (progress = step/total_steps):
    - "linear": α_t = progress
    - "log": α_t = 1 - exp(-progress·scale)      (sobe rápido no início, satura no fim)
    - "exp": α_t = exp((progress-1)·scale)        (sobe devagar no início, rápido no fim)
    """
    progress = min(step / max(total_steps, 1), 1.0)
    if schedule == "linear":
        alpha_t = progress
    elif schedule == "log":
        alpha_t = 1 - np.exp(-progress * scale)
    elif schedule == "exp":
        alpha_t = np.exp((progress - 1) * scale)
    else:
        raise ValueError("tsa_threshold: schedule deve ser 'linear', 'log' ou 'exp'.")

    return alpha_t * (1 - 1.0 / num_classes) + 1.0 / num_classes


def train_model_with_uda(
    model, labeled_loader, unlabeled_loader, test_loader, task,
    optimizer, device, epochs, num_classes,
    confidence_threshold=0.8, sharpening_temperature=0.4, lambda_u=1.0,
    use_tsa=True, tsa_schedule="linear",
    writer=None, log_prefix="",
):
    """
    Loop de treino do UDA (Eq. 1 do paper):

        J(θ) = E[-log p_θ(y*|x1)] + λ·E_x2[ E_x̂~q(x̂|x2)[ CE( p_θ̃(y|x2) , p_θ(y|x̂) ) ] ]

    onde θ̃ é uma cópia FIXA (sem gradiente) dos parâmetros atuais — o alvo da consistência nunca
    propaga gradiente de volta. Duas técnicas adicionais do paper, implementadas aqui:

    - Confidence-based masking: a consistency loss só é computada nos exemplos não rotulados cuja
      confiança em p_θ̃(y|x2) (a predição SEM augmentation forte) ultrapassa confidence_threshold
      (0.8 pra CIFAR-10/SVHN no paper).
    - Sharpening: p_θ̃(y|x2) é suavizada com temperatura (sharpening_temperature=0.4 no paper) antes
      de virar alvo da CE — reforça minimização de entropia.
    - TSA (Training Signal Annealing, use_tsa=True): mascara exemplos ROTULADOS cuja confiança na
      classe certa já passa de um threshold crescente ao longo do treino — evita overfitting
      prematuro no pouco rotulado disponível.

    Requer task com atributo .criterion. num_classes é necessário pra TSA (define o threshold
    inicial 1/num_classes).
    """
    if not hasattr(task, "criterion"):
        raise AttributeError("UDA exige uma task com atributo .criterion (ex.: ClassificationTask).")

    total_steps = epochs * len(labeled_loader)
    global_step = 0
    loss = 0.0
    test_metric = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_steps = 0

        unlabeled_iter = iter(unlabeled_loader)
        progress_bar = tqdm(labeled_loader, desc=f"UDA [{epoch + 1}/{epochs}]", unit="batch", leave=False)
        for labeled_batch in progress_bar:
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)

            x_l, y_l = labeled_batch[0].to(device), labeled_batch[1].to(device)
            weak_u, strong_u = unlabeled_batch[0].to(device), unlabeled_batch[1].to(device)

            optimizer.zero_grad()

            outputs_l = model(x_l)
            per_sample_sup_loss = F.cross_entropy(
                outputs_l.reshape(-1, outputs_l.size(-1)), y_l.reshape(-1), reduction="none",
            ).reshape(y_l.shape)
            per_sample_sup_loss = _reduce_sample_dims(per_sample_sup_loss)

            if use_tsa:
                with torch.no_grad():
                    probs_l = torch.softmax(outputs_l, dim=-1)
                    correct_class_conf = torch.gather(
                        probs_l.reshape(-1, probs_l.size(-1)), 1, y_l.reshape(-1, 1)
                    ).reshape(y_l.shape)
                    correct_class_conf = _reduce_sample_dims(correct_class_conf)
                    eta_t = tsa_threshold(global_step, total_steps, num_classes, schedule=tsa_schedule)
                    tsa_mask = (correct_class_conf < eta_t).float()
                l_s = (per_sample_sup_loss * tsa_mask).sum() / tsa_mask.sum().clamp(min=1.0)
            else:
                l_s = per_sample_sup_loss.mean()

            with torch.no_grad():
                weak_logits = model(weak_u)
                weak_probs = torch.softmax(weak_logits, dim=-1)
                per_position_confidence = torch.max(weak_probs, dim=-1).values
                sample_confidence = _reduce_sample_dims(per_position_confidence)
                mask = (sample_confidence >= confidence_threshold).float()
                target_probs = sharpen_with_temperature(weak_logits, sharpening_temperature)

            strong_logits = model(strong_u)
            strong_log_probs = torch.log_softmax(strong_logits, dim=-1)
            per_sample_kl = -(target_probs * strong_log_probs).sum(dim=-1)  # CE(target, strong)
            per_sample_kl = _reduce_sample_dims(per_sample_kl)

            l_u = (mask * per_sample_kl).mean()

            total_loss = l_s + lambda_u * l_u
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            n_steps += 1
            global_step += 1
            progress_bar.set_postfix(l_s=f"{l_s.item():.3f}", l_u=f"{l_u.item():.3f}",
                                      mask_rate=f"{mask.mean().item():.2f}")

        loss = running_loss / n_steps
        test_metric = _evaluate(model, test_loader, task, device, epoch, epochs)
        print(f"Epoch {epoch + 1}/{epochs}: Loss {loss:.4f} | Test Metric: {test_metric:.4f}")

        if writer is not None:
            writer.add_scalar(f"{log_prefix}Loss/train", loss, epoch)
            writer.add_scalar(f"{log_prefix}Metric/test", test_metric, epoch)

    return loss, test_metric


def _evaluate(model, loader, task, device, epoch, epochs):
    model.eval()
    all_metrics = []
    progress_bar = tqdm(loader, desc=f"Eval [{epoch + 1}/{epochs}]", unit="batch", leave=False)
    for batch in progress_bar:
        all_metrics.extend(task.compute_metric(model, batch, device))
    return float(np.mean(all_metrics))
