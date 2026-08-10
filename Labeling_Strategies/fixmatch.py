import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# FixMatch (Sohn, Berthelot, Li, Zhang, Carlini, Cubuk, Kurakin, Zhang & Raffel, 2020, NeurIPS,
# "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence"): combina
# consistency regularization com pseudo-labeling. Diferente de Labeling_Strategies.labeling_strategies
# (que decidem quais amostras aceitar e devolvem um pseudo-rótulo estático), FixMatch é uma técnica
# de TREINO — o pseudo-rótulo é recomputado a cada passo, a partir da predição do modelo mais atual
# sobre a versão FRACAMENTE aumentada da imagem, e usado como alvo pra classificar a versão
# FORTEMENTE aumentada. Por isso tem seu próprio loop de treino (substitui Training/train.py), como
# LPL/VAAL/WAAL em Query_Strategies.
#
# unlabeled_loader precisa ser construído sobre Labeling_Strategies.augmentation.WeakStrongAugmentDataset
# (batches (weak_image, strong_image, ...)); labeled_loader é um DataLoader normal, com a MESMA
# augmentation fraca aplicada (o paper usa flip+shift pros dois).


def _reduce_sample_dims(tensor):
    while tensor.dim() > 1:
        tensor = tensor.mean(dim=-1)
    return tensor


def train_model_with_fixmatch(
    model, labeled_loader, unlabeled_loader, test_loader, task,
    optimizer, device, epochs, confidence_threshold=0.95, lambda_u=1.0,
    writer=None, log_prefix="",
):
    """
    Loop de treino do FixMatch. A cada passo, pareia um batch rotulado (augmentation fraca) com um
    batch não rotulado (par fraco/forte de Labeling_Strategies.augmentation.WeakStrongAugmentDataset
    — reciclando o loader menor quando necessário). Duas losses (Eqs. 2-4 do paper):

        ℓ_s = (1/B) Σ H(y_b, p_m(y|α(x_b)))                                            — supervisionada
        ℓ_u = (1/μB) Σ 1(max(q_b) ≥ τ) · H(q̂_b, p_m(y|A(u_b))),  q_b = p_m(y|α(u_b))    — pseudo-rótulo

    onde α é augmentation fraca, A é forte, q̂_b=argmax(q_b) é o pseudo-rótulo (hard label). Loss
    final: ℓ_s + λ_u·ℓ_u — sem ramp-up de λ_u (o paper mostra que não é necessário para FixMatch).

    Requer task com atributo .criterion (ex.: ClassificationTask) — a loss supervisionada usa
    diretamente task.criterion(outputs, targets), já que FixMatch assume cross-entropy padrão.
    """
    if not hasattr(task, "criterion"):
        raise AttributeError("FixMatch exige uma task com atributo .criterion (ex.: ClassificationTask).")

    loss = 0.0
    test_metric = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_mask_rate = 0.0
        n_steps = 0

        unlabeled_iter = iter(unlabeled_loader)
        progress_bar = tqdm(labeled_loader, desc=f"FixMatch [{epoch + 1}/{epochs}]", unit="batch", leave=False)
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
            l_s = task.criterion(outputs_l, y_l)

            with torch.no_grad():
                weak_probs = torch.softmax(model(weak_u), dim=-1)  # [Batch, ..., C]
                per_position_confidence, pseudo_labels = torch.max(weak_probs, dim=-1)
                sample_confidence = _reduce_sample_dims(per_position_confidence)
                mask = (sample_confidence >= confidence_threshold).float()

            strong_outputs = model(strong_u)  # [Batch, ..., C]
            per_sample_ce = F.cross_entropy(
                strong_outputs.reshape(-1, strong_outputs.size(-1)), pseudo_labels.reshape(-1),
                reduction="none",
            ).reshape(pseudo_labels.shape)
            per_sample_ce = _reduce_sample_dims(per_sample_ce)

            l_u = (mask * per_sample_ce).mean()  # (1/B) Σ mask·H, equivalente à Eq. 4 do paper

            total_loss = l_s + lambda_u * l_u
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_mask_rate += mask.mean().item()
            n_steps += 1
            progress_bar.set_postfix(l_s=f"{l_s.item():.3f}", l_u=f"{l_u.item():.3f}",
                                      mask_rate=f"{mask.mean().item():.2f}")

        loss = running_loss / n_steps
        mask_rate = running_mask_rate / n_steps
        test_metric = _evaluate(model, test_loader, task, device, epoch, epochs)
        print(f"Epoch {epoch + 1}/{epochs}: Loss {loss:.4f} | Mask Rate {mask_rate:.2f} | "
              f"Test Metric: {test_metric:.4f}")

        if writer is not None:
            writer.add_scalar(f"{log_prefix}Loss/train", loss, epoch)
            writer.add_scalar(f"{log_prefix}MaskRate/train", mask_rate, epoch)
            writer.add_scalar(f"{log_prefix}Metric/test", test_metric, epoch)

    return loss, test_metric


def _evaluate(model, loader, task, device, epoch, epochs):
    model.eval()
    all_metrics = []
    progress_bar = tqdm(loader, desc=f"Eval [{epoch + 1}/{epochs}]", unit="batch", leave=False)
    for batch in progress_bar:
        all_metrics.extend(task.compute_metric(model, batch, device))
    return float(np.mean(all_metrics))
