import copy
import torch
from torch.utils.data import Dataset

# Utilitários compartilhados por várias estratégias de pseudo-labeling baseadas em consistency
# regularization (FixMatch, UDA, MixMatch, ReMixMatch, Mean Teacher) — pra não duplicar a mesma
# lógica de augmentation/sharpening/mixup/EMA em cada arquivo.


class WeakStrongAugmentDataset(Dataset):
    """
    Envolve um dataset base pra devolver DUAS versões da mesma imagem — uma com augmentation fraca
    (flip/shift) e outra com augmentation forte (ex.: RandAugment) — convenção usada por FixMatch e
    UDA (Sohn et al., 2020; Xie et al., 2020). weak_transform/strong_transform recebem a imagem PIL
    original (aplique-os ANTES de qualquer ToTensor/Normalize já embutido no dataset base — ou seja,
    o dataset base deve devolver a imagem crua/PIL, não já tensorizada).

    Devolve (weak_image, strong_image, label, ...demais campos do item original).
    """

    def __init__(self, base_dataset, weak_transform, strong_transform):
        self.base_dataset = base_dataset
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, *rest = self.base_dataset[idx]
        weak = self.weak_transform(image)
        strong = self.strong_transform(image)
        return (weak, strong) + tuple(rest)


class MultiAugmentDataset(Dataset):
    """
    Envolve um dataset base pra devolver K versões aumentadas (mesma transform, reamostrada K vezes)
    da mesma imagem — convenção usada por MixMatch (label guessing, Berthelot et al., 2019) e pelo
    "Augmentation Anchoring" do ReMixMatch (Berthelot et al., 2019b).

    Devolve (lista_de_K_imagens_aumentadas, label, ...demais campos do item original).
    """

    def __init__(self, base_dataset, transform, k=2):
        self.base_dataset = base_dataset
        self.transform = transform
        self.k = k

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, *rest = self.base_dataset[idx]
        augmented = [self.transform(image) for _ in range(self.k)]
        return (augmented,) + tuple(rest)


def sharpen(probs, temperature):
    """
    Sharpen(p,T)_i = p_i^(1/T) / Σ_j p_j^(1/T)  (Eq. 7 do MixMatch, Berthelot et al., 2019) — reduz a
    entropia da distribuição categórica probs; T→0 se aproxima de um one-hot (Dirac), T=1 é
    identidade. Usado como alvo pra minimização de entropia implícita em MixMatch/ReMixMatch/UDA.
    """
    sharpened = probs ** (1.0 / temperature)
    return sharpened / sharpened.sum(dim=-1, keepdim=True)


def mixup(x1, x2, p1, p2, alpha):
    """
    MixUp modificado do MixMatch (Eqs. 8-11): λ ~ Beta(α,α); λ' = max(λ, 1-λ) — garante que o
    resultado fique mais perto de x1 (preserva a ordem do par, necessário pro MixMatch porque batches
    rotulado/não-rotulado são concatenados e depois misturados com pares aleatórios).

        x' = λ'·x1 + (1-λ')·x2,   p' = λ'·p1 + (1-λ')·p2

    x1/x2: tensores de entrada (imagens). p1/p2: rótulos/probabilidades (mesmo formato, one-hot ou
    distribuição). Devolve (x', p').
    """
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    lam = max(lam, 1 - lam)
    x_mixed = lam * x1 + (1 - lam) * x2
    p_mixed = lam * p1 + (1 - lam) * p2
    return x_mixed, p_mixed


def ema_update(ema_model, model, decay):
    """
    Atualização por média móvel exponencial dos pesos: θ'_t = decay·θ'_{t-1} + (1-decay)·θ_t (Mean
    Teacher, Tarvainen & Valpola, 2017, Eq. sem número — "EMA of successive θ weights"). ema_model é
    atualizado in-place; model não é modificado. Chame depois de cada optimizer.step() do model.
    """
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.mul_(decay).add_(param.detach(), alpha=1 - decay)
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


def clone_model(model):
    """Cria uma cópia independente (pesos + buffers) do modelo, pra usar como teacher/EMA inicial."""
    return copy.deepcopy(model)
