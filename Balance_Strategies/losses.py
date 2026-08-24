import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Diferente do restante de Balance_Strategies (que decide QUAIS/QUANTOS candidatos entram numa
# seleção), este módulo cobre a família "feature representation" da survey de desbalanceamento: muda
# COMO a loss de treino pondera cada classe/amostra, sem mexer em quantas amostras de cada classe
# existem. Usadas como `criterion` de uma Task (Training/tasks.py), ex.:
# ClassificationTask(criterion=FocalLoss(gamma=2.0)).


def compute_class_weights(labels, num_classes, scheme="inverse_frequency"):
    """
    Cost-sensitive learning / class weights (Ling & Sheng, 2008; He & Garcia, 2009): pesos por classe
    pra usar em nn.CrossEntropyLoss(weight=...), corrigindo o viés do modelo em favor da classe
    majoritária.

    labels: array/tensor 1D com o rótulo de cada amostra do conjunto de treino (já "achatado" pra um
    índice inteiro por posição — pra saídas multi-posição, achate antes de chamar).
    scheme:
    - "inverse_frequency": w_c = N / (num_classes · n_c) — quanto menos amostras a classe c tem, maior
      o peso (média dos pesos ponderada pela frequência real = 1).
    - "inverse_sqrt": w_c = 1/sqrt(n_c), normalizado pra média 1 — correção mais suave, útil quando
      inverse_frequency super-corrige (comum com desbalanceamento extremo).
    """
    labels = np.asarray(labels).flatten()
    counts = np.array([max(np.sum(labels == c), 1) for c in range(num_classes)], dtype=np.float64)

    if scheme == "inverse_frequency":
        weights = labels.size / (num_classes * counts)
    elif scheme == "inverse_sqrt":
        weights = 1.0 / np.sqrt(counts)
        weights = weights * (num_classes / weights.sum())
    else:
        raise ValueError(f"scheme desconhecido: {scheme!r} (use 'inverse_frequency' ou 'inverse_sqrt').")

    return torch.tensor(weights, dtype=torch.float32)


def class_balanced_weights(class_counts, beta=0.9999):
    """
    Class-Balanced Loss (Cui, Jia, Lin, Song & Belongie, 2019, CVPR, "Class-Balanced Loss Based on
    Effective Number of Samples"): em vez de pesar por 1/n_c (inverse frequency) direto, usa o
    "número efetivo de amostras" — a ideia é que amostras adicionais de uma classe já bem
    representada se sobrepõem (redundância), então o benefício marginal de cada uma diminui:

        E_{n_c} = (1 - beta^{n_c}) / (1 - beta),      w_c = (1 - beta) / (1 - beta^{n_c}) = 1/E_{n_c}

    beta in [0,1) controla o quão agressivamente classes com muitas amostras são sub-ponderadas:
    beta=0 -> sem reponderação (w_c=1 pra todas); beta->1 -> se aproxima de 1/n_c (inverse frequency
    pura). O paper usa beta em {0.9, 0.99, 0.999, 0.9999} dependendo do dataset.

    class_counts: array/lista com o número de amostras de cada classe (n_c). Devolve um tensor de
    pesos normalizados pra média 1 (mesma convenção da implementação oficial dos autores), pronto pra
    usar como `alpha` de FocalLoss ou `weight` de nn.CrossEntropyLoss.
    """
    counts = np.asarray(class_counts, dtype=np.float64)
    effective_num = 1.0 - np.power(beta, counts)
    weights = (1.0 - beta) / np.maximum(effective_num, 1e-8)
    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin, Goyal, Girshick, He & Dollár, 2017, ICCV, "Focal Loss for Dense Object
    Detection"):

        FL(p_t) = -alpha_t · (1 - p_t)^gamma · log(p_t)

    onde p_t é a probabilidade prevista pra classe verdadeira. O fator (1-p_t)^gamma reduz a
    contribuição de exemplos já bem classificados (p_t alto), deixando o gradiente do treino focar
    nos exemplos difíceis — amostras de classes minoritárias tendem a ser mais difíceis em média,
    então se beneficiam relativamente mais dessa reponderação, mesmo sem alpha nenhum.

    gamma: fator de modulação (paper usa 2.0; gamma=0 recupera a cross-entropy comum).
    alpha: peso por classe, opcional (escalar ou tensor de tamanho num_classes, ex. saído de
    compute_class_weights/class_balanced_weights) — combina focal loss com cost-sensitive learning.
    """

    def __init__(self, gamma=2.0, alpha=None, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        # outputs: [Batch, ..., C]  targets: [Batch, ...]
        outputs_flat = outputs.reshape(-1, outputs.size(-1))
        targets_flat = targets.reshape(-1)

        log_probs = F.log_softmax(outputs_flat, dim=-1)
        # -log(p_t) SEM peso de classe — p_t precisa vir daqui, não de uma CE já ponderada por alpha,
        # senão exp(-ce) deixa de ser p_t e passa a ser p_t^alpha_t.
        ce = F.nll_loss(log_probs, targets_flat, ignore_index=self.ignore_index, reduction="none")
        p_t = torch.exp(-ce)

        loss = ((1 - p_t) ** self.gamma) * ce

        if self.alpha is not None:
            alpha = self.alpha if torch.is_tensor(self.alpha) else torch.tensor(self.alpha)
            alpha = alpha.to(outputs.device)
            alpha_t = alpha[targets_flat.clamp(min=0)]  # clamp evita indexar com ignore_index negativo
            loss = alpha_t * loss

        valid = targets_flat != self.ignore_index
        return loss[valid].mean() if valid.any() else loss.sum() * 0.0


# --- DYNAMIC CURRICULUM LEARNING (Wang, Gan, Yang, Wu & Yan, 2019, ICCV, "Dynamic Curriculum
# Learning for Imbalanced Data Classification") ---
#
# O paper propõe dois componentes: (1) um Sampling Scheduler + Dynamic Selective Learning (DSL) loss
# — a parte diretamente ligada a balanceamento, implementada abaixo — e (2) uma loss auxiliar de
# metric learning (Triplet Loss with Easy Anchors), pensada pra melhorar a qualidade do embedding
# aprendido. O item (2) fica FORA do escopo aqui por ser uma técnica de aprendizado de representação
# (metric learning), ortogonal a "balanceamento" propriamente dito — só o núcleo (1) foi implementado.

def scheduler_cos(l, total_epochs):
    """SF_cos (Eq. 1): decai rápido no início, devagar no fim — aprendizado começa rápido."""
    return float(np.cos((l / total_epochs) * (np.pi / 2)))


def scheduler_linear(l, total_epochs):
    """SF_linear (Eq. 2): decaimento constante."""
    return 1.0 - l / total_epochs


def scheduler_exp(l, total_epochs, lam=0.9):
    """SF_exp (Eq. 3): decai devagar no início, rápido no fim — aprendizado começa devagar."""
    return float(lam ** l)


def scheduler_composite(l, total_epochs):
    """SF_composite (Eq. 4): rápido, depois devagar, depois rápido de novo — usado como exemplo
    principal no paper (Eq. 11)."""
    return float(0.5 * np.cos((l / total_epochs) * np.pi) + 0.5)


class DynamicCurriculumLoss(nn.Module):
    """
    Sampling Scheduler + Dynamic Selective Learning loss (Eqs. 1-8 do paper): a cada época l (de um
    total de total_epochs), a distribuição-alvo de cada classe no batch é a distribuição real do
    dataset de treino elevada a uma função de scheduler g(l) que decai de 1 (distribuição real,
    desbalanceada) pra 0 (distribuição balanceada, já que qualquer proporção^0 = 1):

        D_target(l) = D_train ^ g(l)                                              (Eq. 6)

    Comparando D_target(l) com a distribuição REAL do batch atual (D_current), cada classe j recebe
    um peso por amostra (Eq. 8):

    - Se D_target,j(l)/D_current,j >= 1 (classe rara NESSE batch): todas as amostras da classe
      recebem peso = essa razão (upweight).
    - Se < 1 (classe super-representada NESSE batch): cada amostra da classe é mantida com peso 1 com
      probabilidade = essa razão, e descartada (peso 0) caso contrário — reamostragem estocástica.

    class_counts: contagem de amostras por classe no dataset de treino INTEIRO (define D_train, Eq. 5).
    scheduler_fn: uma de scheduler_cos/linear/exp/composite (padrão: composite, o exemplo do paper).
    base_criterion: loss por amostra a ponderar (padrão: cross-entropy); precisa aceitar
    reduction="none" e devolver um valor por amostra.

    Requer chamar .set_epoch(epoch) a cada época de treino (train_model não sabe disso sozinho —
    quem orquestra o loop, ex. um wrapper em torno de Training/train.py, precisa chamar).
    """

    def __init__(self, class_counts, total_epochs, scheduler_fn=scheduler_composite, base_criterion=None):
        super().__init__()
        counts = np.asarray(class_counts, dtype=np.float64)
        self.class_distribution = counts / counts.min()  # D_train (Eq. 5)
        self.total_epochs = total_epochs
        self.scheduler_fn = scheduler_fn
        self.base_criterion = base_criterion or nn.CrossEntropyLoss(reduction="none")
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, outputs, targets):
        outputs_flat = outputs.reshape(-1, outputs.size(-1))
        targets_flat = targets.reshape(-1)
        num_classes = outputs_flat.size(-1)
        device = outputs.device

        g_l = self.scheduler_fn(self.current_epoch, self.total_epochs)
        target_distribution = torch.tensor(self.class_distribution, dtype=torch.float32, device=device) ** g_l

        current_counts = torch.bincount(targets_flat, minlength=num_classes).float()
        # min() só entre classes PRESENTES no batch — clamp(min=1) sobre o tensor inteiro antes do
        # min() faria uma classe ausente do batch (contagem 0 -> vira 1) quase sempre virar o mínimo
        # artificialmente, colapsando a normalização e distorcendo a razão da Eq. 8 para todas as
        # classes.
        present_counts = current_counts[current_counts > 0]
        min_count = present_counts.min() if present_counts.numel() > 0 else torch.tensor(1.0, device=device)
        current_distribution = current_counts / min_count.clamp(min=1)

        sample_weights = torch.ones_like(targets_flat, dtype=torch.float32)
        for c in range(num_classes):
            class_mask = targets_flat == c
            n_in_class = int(class_mask.sum().item())
            if n_in_class == 0:
                continue

            ratio = (target_distribution[c] / current_distribution[c].clamp(min=1e-8)).item()
            if ratio >= 1:
                sample_weights[class_mask] = ratio
            else:
                keep = torch.rand(n_in_class, device=device) < ratio
                class_weights = torch.zeros(n_in_class, device=device)
                class_weights[keep] = 1.0
                sample_weights[class_mask] = class_weights

        per_sample_loss = self.base_criterion(outputs_flat, targets_flat)  # -log p(y|x), sem peso ainda
        weighted_loss = sample_weights * per_sample_loss

        # Eq. 7 do paper: L_DSL = -(1/N) Σ w·log p(...), com N = TAMANHO FIXO DO BATCH — não a
        # contagem de amostras não-descartadas. Amostras com sample_weights=0 (descartadas na
        # reamostragem estocástica) já contribuem 0 à soma, então dividir por N direto (em vez de só
        # pelas "válidas") é a normalização correta; dividir pelas válidas infla a loss quanto mais
        # amostras forem descartadas no batch, o que não corresponde à fórmula do paper.
        return weighted_loss.sum() / targets_flat.numel()
