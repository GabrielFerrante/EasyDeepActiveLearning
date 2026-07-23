import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou


# --- MÉTRICAS PADRÃO (todas devolvem uma lista/array com um valor por amostra do batch) ---

def classification_accuracy(outputs, targets):
    preds = torch.argmax(outputs, dim=-1)
    correct = (preds == targets).float()
    while correct.dim() > 1:
        correct = torch.mean(correct, dim=-1)
    return correct.cpu().tolist()


def per_image_mean_iou(outputs, targets, eps=1e-6):
    preds = torch.argmax(outputs, dim=1)  # [B, H, W]
    results = []
    for pred, target in zip(preds, targets):
        ious = []
        for c in torch.unique(target):
            pred_c = pred == c
            target_c = target == c
            intersection = (pred_c & target_c).sum().float()
            union = (pred_c | target_c).sum().float()
            if union > 0:
                ious.append((intersection / (union + eps)).item())
        results.append(sum(ious) / len(ious) if ious else 0.0)
    return results


def detection_recall_at_iou(predictions, targets, iou_threshold=0.5, score_threshold=0.5):
    """
    Métrica simplificada (não é a mAP do COCO): fração de caixas verdadeiras casadas por alguma
    predição acima do score_threshold com IoU >= iou_threshold. Para avaliação rigorosa, passe seu
    próprio metric_fn (ex.: torchmetrics.detection.MeanAveragePrecision) ao instanciar DetectionTask.
    """
    recalls = []
    for pred, target in zip(predictions, targets):
        gt_boxes = target['boxes']
        if gt_boxes.numel() == 0:
            recalls.append(1.0)  # nada pra achar nessa imagem, considera acerto trivial
            continue
        keep = pred['scores'] >= score_threshold
        pred_boxes = pred['boxes'][keep]
        if pred_boxes.numel() == 0:
            recalls.append(0.0)
            continue
        best_iou_per_gt = box_iou(gt_boxes, pred_boxes).max(dim=1).values
        recalls.append((best_iou_per_gt >= iou_threshold).float().mean().item())
    return recalls


def retrieval_top1_accuracy(image_embeds, text_embeds):
    sims = image_embeds @ text_embeds.T
    preds = torch.argmax(sims, dim=1)
    targets = torch.arange(sims.size(0), device=sims.device)
    return (preds == targets).float().cpu().tolist()


# --- TASKS ---

class Task:
    """
    Encapsula as convenções de batch/forward de uma família de tarefa, para que Training/train.py e
    as Query_Strategies (estratégia de incerteza) funcionem com qualquer uma sem precisar conhecer o
    formato de saída do modelo.
    """

    def compute_loss(self, model, batch, device):
        raise NotImplementedError

    def compute_metric(self, model, batch, device):
        """Deve devolver uma lista/array com um valor de métrica por amostra do batch."""
        raise NotImplementedError

    def compute_uncertainty(self, model, batch, device):
        """Deve devolver uma lista/array com um score de incerteza por amostra (maior = mais incerto)."""
        raise NotImplementedError


class ClassificationTask(Task):
    """Supervisionada: batch = (images, labels, ...). Cobre também saídas com múltiplas posições/heads
    (ex.: [Batch, N, num_classes]), como no exemplo SVHN."""

    def __init__(self, criterion=None, metric_fn=None):
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.metric_fn = metric_fn or classification_accuracy

    def compute_loss(self, model, batch, device):
        inputs, targets = batch[0].to(device), batch[1].to(device)
        outputs = model(inputs)
        return self.criterion(outputs, targets)

    def compute_metric(self, model, batch, device):
        inputs, targets = batch[0].to(device), batch[1].to(device)
        with torch.no_grad():
            outputs = model(inputs)
        return self.metric_fn(outputs, targets)

    def compute_uncertainty(self, model, batch, device):
        inputs = batch[0].to(device)
        with torch.no_grad():
            outputs = model(inputs)  # [Batch, ..., num_classes]

        probs = torch.softmax(outputs, dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        while entropy.dim() > 1:
            entropy = torch.mean(entropy, dim=-1)

        return entropy.cpu().numpy()


class SegmentationTask(Task):
    """Supervisionada, pixel a pixel: batch = (images, masks); modelo devolve [Batch, C, H, W]."""

    def __init__(self, criterion=None, metric_fn=None):
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.metric_fn = metric_fn or per_image_mean_iou

    def compute_loss(self, model, batch, device):
        inputs, targets = batch[0].to(device), batch[1].to(device)
        outputs = model(inputs)
        return self.criterion(outputs, targets)

    def compute_metric(self, model, batch, device):
        inputs, targets = batch[0].to(device), batch[1].to(device)
        with torch.no_grad():
            outputs = model(inputs)
        return self.metric_fn(outputs, targets)

    def compute_uncertainty(self, model, batch, device):
        inputs = batch[0].to(device)
        with torch.no_grad():
            outputs = model(inputs)  # [Batch, C, H, W]

        probs = torch.softmax(outputs, dim=1)
        log_probs = torch.log(probs + 1e-10)
        entropy_map = -torch.sum(probs * log_probs, dim=1)  # [Batch, H, W]
        entropy = torch.mean(entropy_map, dim=(1, 2))  # incerteza média por pixel, por amostra

        return entropy.cpu().numpy()


def _move_images(images, device):
    return [img.to(device) for img in images]


def _move_targets(targets, device):
    return [{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets]


class DetectionTask(Task):
    """
    Segue a convenção de detecção do torchvision (FasterRCNN/RetinaNet/SSD/...): batch = (images, targets),
    onde images é uma lista de tensores (tamanhos variáveis) e targets uma lista de dicts com pelo menos
    'boxes' e 'labels' — normalmente via um collate_fn customizado no DataLoader. Em model.train(), o
    próprio model(images, targets) devolve um dict de losses (não precisa de criterion externo); em
    model.eval(), model(images) devolve as predições (dicts com 'boxes', 'labels', 'scores').
    """

    def __init__(self, metric_fn=None):
        self.metric_fn = metric_fn or detection_recall_at_iou

    def compute_loss(self, model, batch, device):
        images = _move_images(batch[0], device)
        targets = _move_targets(batch[1], device)

        loss_dict = model(images, targets)  # model.training deve estar True (ver Training/train.py)
        return sum(loss_dict.values())

    def compute_metric(self, model, batch, device):
        images = _move_images(batch[0], device)
        targets = _move_targets(batch[1], device)

        with torch.no_grad():
            predictions = model(images)  # model.training deve estar False
        return self.metric_fn(predictions, targets)

    def compute_uncertainty(self, model, batch, device):
        images = _move_images(batch[0], device)

        with torch.no_grad():
            predictions = model(images)

        uncertainties = []
        for pred in predictions:
            scores = pred['scores']
            # Sem detecção nenhuma = o modelo não achou nada em que confiar: incerteza máxima
            uncertainties.append(1.0 if scores.numel() == 0 else 1.0 - scores.mean().item())

        return uncertainties


class VLMContrastiveTask(Task):
    """
    Treino contrastivo estilo CLIP: batch = (images, texts) — pares alinhados (texts pode já vir
    tokenizado ou cru, dependendo do que model.forward espera). O modelo deve implementar
    forward(images, texts) -> (image_embeds, text_embeds); a task normaliza os embeddings e aplica a
    temperatura. Não existe "rótulo" no sentido supervisionado clássico: a loss é InfoNCE simétrica
    entre imagem e texto dentro do próprio batch.
    """

    def __init__(self, temperature=0.07, metric_fn=None):
        self.temperature = temperature
        self.metric_fn = metric_fn or retrieval_top1_accuracy

    def _embed(self, model, batch, device):
        images, texts = batch[0], batch[1]
        images = images.to(device)
        if torch.is_tensor(texts):
            texts = texts.to(device)

        image_embeds, text_embeds = model(images, texts)
        return F.normalize(image_embeds, dim=-1), F.normalize(text_embeds, dim=-1)

    def compute_loss(self, model, batch, device):
        image_embeds, text_embeds = self._embed(model, batch, device)
        logits = image_embeds @ text_embeds.T / self.temperature
        targets = torch.arange(logits.size(0), device=device)

        loss_i2t = F.cross_entropy(logits, targets)
        loss_t2i = F.cross_entropy(logits.T, targets)
        return (loss_i2t + loss_t2i) / 2

    def compute_metric(self, model, batch, device):
        with torch.no_grad():
            image_embeds, text_embeds = self._embed(model, batch, device)
        return self.metric_fn(image_embeds, text_embeds)

    def compute_uncertainty(self, model, batch, device):
        """
        Incerteza = margem pequena entre o texto mais parecido e o segundo mais parecido, calculada
        dentro do próprio batch. É uma aproximação local e barata, não uma medida global — o pool não
        rotulado inteiro não cabe na comparação de similaridade a cada batch.
        """
        with torch.no_grad():
            image_embeds, text_embeds = self._embed(model, batch, device)

        sims = image_embeds @ text_embeds.T  # [Batch, Batch]
        k = min(2, sims.size(1))
        top_k = torch.topk(sims, k=k, dim=1).values
        margin = top_k[:, 0] - top_k[:, 1] if k > 1 else top_k[:, 0]

        return (-margin).cpu().numpy()  # margem pequena => mais incerto
