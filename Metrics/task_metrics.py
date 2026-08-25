import numpy as np
import torch
from torchvision.ops import box_iou

from tasks import ClassificationTask, RegressionTask, SegmentationTask, DetectionTask, VLMContrastiveTask

# Diferente de Task.compute_metric (Training/tasks.py) — que devolve um valor por amostra de UM
# batch, pra uso leve dentro de um loop de treino (ex.: a métrica que train_model imprime a cada
# época) — este módulo calcula um RELATÓRIO agregado sobre a coleção INTEIRA de predições de um
# loader (precision/recall/F1, matriz de confusão, mAP simplificada, R², recall@k, ...): métricas que
# não podem ser calculadas como média de valores por amostra, precisam ver o conjunto todo de uma vez.
#
# evaluate_task(model, loader, task, device) é o ponto de entrada único: identifica o tipo de task e
# devolve o relatório certo. Cada relatório também pode ser chamado direto se você já tem as predições
# acumuladas (ex.: vindas de fora do fluxo de treino da lib).


# --- CLASSIFICAÇÃO ---

def confusion_matrix(preds, targets, num_classes):
    """Matriz de confusão [num_classes, num_classes], linha = classe verdadeira, coluna = predita."""
    preds = preds.reshape(-1).long()
    targets = targets.reshape(-1).long()
    idx = targets * num_classes + preds
    cm = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm.numpy()


def _precision_recall_f1(cm):
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)   # contagem real por classe (soma da linha)
    predicted = cm.sum(axis=0).astype(np.float64)  # contagem predita por classe (soma da coluna)

    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return precision, recall, f1, support


def top_k_accuracy(outputs, targets, k=5):
    """Fração de amostras cujo rótulo verdadeiro está entre as k classes de maior probabilidade."""
    k = min(k, outputs.size(-1))
    top_k_preds = outputs.reshape(-1, outputs.size(-1)).topk(k, dim=-1).indices  # [N, k]
    correct = (top_k_preds == targets.reshape(-1, 1)).any(dim=-1).float()
    return float(correct.mean().item())


def classification_report(outputs, targets, num_classes=None):
    """
    outputs: logits [N, ..., C] (dimensões extras já achatadas por collect_predictions, se houver).
    targets: rótulos [N, ...], mesma forma de outputs sem a dimensão de classe.
    """
    outputs = outputs.detach()
    targets = targets.detach().long()
    preds = outputs.argmax(dim=-1)
    num_classes = num_classes or outputs.size(-1)

    cm = confusion_matrix(preds, targets, num_classes)
    precision, recall, f1, support = _precision_recall_f1(cm)
    present = support > 0
    total_support = support.sum()

    accuracy = float((preds.reshape(-1) == targets.reshape(-1)).float().mean().item())
    macro_precision = float(precision[present].mean()) if present.any() else 0.0
    macro_recall = float(recall[present].mean()) if present.any() else 0.0
    macro_f1 = float(f1[present].mean()) if present.any() else 0.0
    weighted_f1 = float((f1 * support).sum() / total_support) if total_support > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision_macro": macro_precision,
        "recall_macro": macro_recall,
        "f1_macro": macro_f1,
        "f1_weighted": weighted_f1,
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": support.astype(int).tolist(),
        "confusion_matrix": cm.tolist(),
    }


# --- REGRESSÃO ---

def regression_report(outputs, targets):
    """MAE, MSE, RMSE, R² e MAPE (ignorando alvos ~0, onde MAPE não é definido)."""
    outputs = outputs.detach().float().reshape(-1)
    targets = targets.detach().float().reshape(-1)
    error = outputs - targets

    mae = error.abs().mean()
    mse = (error ** 2).mean()
    rmse = mse.sqrt()

    ss_res = (error ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = (1 - ss_res / ss_tot) if ss_tot > 0 else torch.tensor(float("nan"))

    nonzero = targets.abs() > 1e-8
    mape = (error[nonzero] / targets[nonzero]).abs().mean() * 100 if nonzero.any() else torch.tensor(float("nan"))

    return {
        "mae": float(mae.item()),
        "mse": float(mse.item()),
        "rmse": float(rmse.item()),
        "r2": float(r2.item()),
        "mape": float(mape.item()),
    }


# --- SEGMENTAÇÃO ---

def segmentation_report(batched_outputs, batched_targets, num_classes=None):
    """
    batched_outputs: lista de tensores de logits [B,C,H,W], um por batch (a saída de
    collect_predictions, acumulada por evaluate_task). batched_targets: lista de máscaras [B,H,W].
    """
    if num_classes is None:
        num_classes = batched_outputs[0].size(1)

    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)
    correct_pixels, total_pixels = 0, 0
    per_image_ious = []

    for outputs, targets in zip(batched_outputs, batched_targets):
        preds = outputs.argmax(dim=1)  # [B,H,W]
        correct_pixels += (preds == targets).sum().item()
        total_pixels += targets.numel()

        for pred, target in zip(preds, targets):
            ious = []
            for c in range(num_classes):
                pred_c, target_c = pred == c, target == c
                inter = (pred_c & target_c).sum().item()
                uni = (pred_c | target_c).sum().item()
                intersection[c] += inter
                union[c] += uni
                if uni > 0:
                    ious.append(inter / uni)
            per_image_ious.append(sum(ious) / len(ious) if ious else 0.0)

    iou_per_class = np.divide(intersection, union, out=np.zeros(num_classes), where=union > 0)
    # Dice = 2|A∩B| / (|A|+|B|); como |A|+|B| = intersection + union, Dice = 2*inter/(inter+union).
    denom = intersection + union
    dice_per_class = np.divide(2 * intersection, denom, out=np.zeros(num_classes), where=denom > 0)
    present = union > 0

    return {
        "pixel_accuracy": correct_pixels / total_pixels if total_pixels > 0 else 0.0,
        "mean_iou": float(iou_per_class[present].mean()) if present.any() else 0.0,
        "mean_dice": float(dice_per_class[present].mean()) if present.any() else 0.0,
        "iou_per_class": iou_per_class.tolist(),
        "dice_per_class": dice_per_class.tolist(),
        "mean_iou_per_image": float(np.mean(per_image_ious)) if per_image_ious else 0.0,
    }


# --- DETECÇÃO ---

def detection_report(predictions, targets, iou_thresholds=(0.5,), score_threshold=0.5):
    """
    Precision/recall/F1 por threshold de IoU, via correspondência gulosa (cada predição, processada
    em ordem decrescente de score, reivindica seu melhor GT ainda livre) — não é a mAP oficial do COCO
    (que usa interpolação de 101 pontos de recall); pra isso, use torchmetrics.detection.
    MeanAveragePrecision. `mean_ap_approx` aqui é só a média das precisions por threshold.
    """
    per_threshold = {}
    for thr in iou_thresholds:
        tp, fp, fn = 0, 0, 0
        for pred, target in zip(predictions, targets):
            gt_boxes = target["boxes"]
            keep = pred["scores"] >= score_threshold
            pred_boxes = pred["boxes"][keep]
            pred_scores = pred["scores"][keep]

            if pred_boxes.numel() == 0:
                fn += gt_boxes.size(0)
                continue

            order = torch.argsort(pred_scores, descending=True)
            pred_boxes = pred_boxes[order]

            if gt_boxes.numel() == 0:
                fp += pred_boxes.size(0)
                continue

            ious = box_iou(pred_boxes, gt_boxes)  # [num_pred, num_gt], pred já ordenado por score
            matched_gt = set()
            for p_idx in range(pred_boxes.size(0)):
                best_gt_idx = int(ious[p_idx].argmax().item())
                best_iou = ious[p_idx, best_gt_idx].item()
                if best_iou >= thr and best_gt_idx not in matched_gt:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1
            fn += gt_boxes.size(0) - len(matched_gt)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_threshold[thr] = {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    mean_ap_approx = float(np.mean([per_threshold[t]["precision"] for t in iou_thresholds]))
    return {"per_iou_threshold": per_threshold, "mean_ap_approx": mean_ap_approx}


# --- VLM / RETRIEVAL ---

def recall_at_k(sims, k):
    """sims: [N,N], similaridade query->target (par correto na diagonal). Fração de queries cujo par
    correto está entre os k targets mais similares."""
    k = min(k, sims.size(1))
    top_k = sims.topk(k, dim=1).indices
    targets = torch.arange(sims.size(0), device=sims.device).unsqueeze(1)
    correct = (top_k == targets).any(dim=1).float()
    return float(correct.mean().item())


def mean_reciprocal_rank(sims):
    targets = torch.arange(sims.size(0), device=sims.device)
    order = sims.argsort(dim=1, descending=True)
    ranks = (order == targets.unsqueeze(1)).float().argmax(dim=1) + 1  # posição do alvo, 1-indexada
    return float((1.0 / ranks.float()).mean().item())


def retrieval_report(image_embeds, text_embeds, ks=(1, 5, 10)):
    sims_i2t = image_embeds @ text_embeds.T
    sims_t2i = text_embeds @ image_embeds.T

    report = {"mrr_i2t": mean_reciprocal_rank(sims_i2t), "mrr_t2i": mean_reciprocal_rank(sims_t2i)}
    for k in ks:
        report[f"recall_at_{k}_i2t"] = recall_at_k(sims_i2t, k)
        report[f"recall_at_{k}_t2i"] = recall_at_k(sims_t2i, k)
    return report


# --- DISPATCHER ---

def evaluate_task(model, loader, task, device):
    """
    Roda o modelo sobre TODO o loader (uma passada de avaliação completa) e devolve o relatório de
    métricas relevante pro tipo de task, via task.collect_predictions (implementado por cada classe
    de Training/tasks.py) — esta função não precisa conhecer o formato de batch de cada tarefa.
    """
    model.eval()

    if isinstance(task, (ClassificationTask, RegressionTask)):
        all_outputs, all_targets = [], []
        for batch in loader:
            outputs, targets = task.collect_predictions(model, batch, device)
            all_outputs.append(outputs)
            all_targets.append(targets)
        outputs_cat = torch.cat(all_outputs, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)
        if isinstance(task, ClassificationTask):
            return classification_report(outputs_cat, targets_cat)
        return regression_report(outputs_cat, targets_cat)

    if isinstance(task, SegmentationTask):
        all_outputs, all_targets = [], []
        for batch in loader:
            outputs, targets = task.collect_predictions(model, batch, device)
            all_outputs.append(outputs)
            all_targets.append(targets)
        return segmentation_report(all_outputs, all_targets)

    if isinstance(task, DetectionTask):
        all_preds, all_targets = [], []
        for batch in loader:
            preds, targets = task.collect_predictions(model, batch, device)
            all_preds.extend(preds)
            all_targets.extend(targets)
        return detection_report(all_preds, all_targets)

    if isinstance(task, VLMContrastiveTask):
        all_img, all_txt = [], []
        for batch in loader:
            img, txt = task.collect_predictions(model, batch, device)
            all_img.append(img)
            all_txt.append(txt)
        return retrieval_report(torch.cat(all_img, dim=0), torch.cat(all_txt, dim=0))

    raise TypeError(f"evaluate_task: tipo de task não suportado: {type(task)}")
