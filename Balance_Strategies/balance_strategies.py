import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors


def _class_key(label):
    """Chave hasheável pra agrupar por classe: funciona com valor escalar ou tensor multi-posição
    (ex.: sequência de dígitos do SVHN, usada como tupla)."""
    value = label.tolist() if hasattr(label, "tolist") else label
    return tuple(value) if isinstance(value, list) else value


def class_balance_strategy(candidates, max_per_class=None):
    """
    Balanceia uma lista de candidatos por classe, mantendo os de maior score primeiro em cada classe.

    candidates: lista de (score, index, label) — ex.: (confiança, índice no dataset, pseudo-rótulo
    previsto), como em Labeling_Strategies.labeling_strategies.pseudo_labeling_strategy. O score só desempata
    a ordem dentro da classe, não decide quem entra.

    Se max_per_class não for informado, cada classe é limitada à contagem da classe com MENOS
    candidatos (balanceamento estrito e automático) — evita que uma classe super-representada (a que
    o modelo mais acerta, logo mais confiante) domine a seleção. Se informado, cada classe é limitada
    a min(contagem_da_classe, max_per_class).

    Devolve a lista de candidatos balanceada, no mesmo formato de entrada (score, index, label).
    """
    if not candidates:
        return []

    candidates_by_class = {}
    for score, index, label in candidates:
        candidates_by_class.setdefault(_class_key(label), []).append((score, index, label))

    quota = max_per_class if max_per_class is not None else min(len(group) for group in candidates_by_class.values())

    balanced = []
    for group in candidates_by_class.values():
        group.sort(key=lambda c: c[0], reverse=True)  # mais confiantes primeiro
        balanced.extend(group[:quota])

    return balanced


# --- RE-AMOSTRAGEM SIMPLES (não precisa de features, só do rótulo) ---

def random_oversample_strategy(candidates, target_count=None, random_state=42):
    """
    Random Oversampling: duplica aleatoriamente candidatos de classes minoritárias até que cada
    classe tenha (aproximadamente) target_count candidatos.

    candidates: lista de (score, index, label) — mesmo formato de class_balance_strategy.
    target_count: se não informado, usa a contagem da classe MAJORITÁRIA (balanceamento completo).

    Diferente das outras estratégias, pode devolver o MESMO candidato repetido (índice duplicado) —
    é esperado: o Subset/DataLoader vai ler essa amostra mais de uma vez.
    """
    if not candidates:
        return []

    rng = np.random.default_rng(random_state)
    candidates_by_class = {}
    for c in candidates:
        candidates_by_class.setdefault(_class_key(c[2]), []).append(c)

    target_count = target_count or max(len(group) for group in candidates_by_class.values())

    balanced = []
    for group in candidates_by_class.values():
        balanced.extend(group)
        if len(group) < target_count:
            extra_idx = rng.integers(0, len(group), size=target_count - len(group))
            balanced.extend(group[i] for i in extra_idx)

    return balanced


def random_undersample_strategy(candidates, target_count=None, random_state=42):
    """
    Random Undersampling: remove aleatoriamente candidatos de classes majoritárias até que cada
    classe tenha (no máximo) target_count candidatos.

    candidates: lista de (score, index, label). target_count: se não informado, usa a contagem da
    classe MINORITÁRIA — mesmo alvo padrão de class_balance_strategy, mas removendo aleatoriamente
    em vez de manter os de maior score.
    """
    if not candidates:
        return []

    rng = np.random.default_rng(random_state)
    candidates_by_class = {}
    for c in candidates:
        candidates_by_class.setdefault(_class_key(c[2]), []).append(c)

    target_count = target_count or min(len(group) for group in candidates_by_class.values())

    balanced = []
    for group in candidates_by_class.values():
        if len(group) <= target_count:
            balanced.extend(group)
        else:
            chosen = rng.choice(len(group), size=target_count, replace=False)
            balanced.extend(group[i] for i in chosen)

    return balanced


# --- RE-AMOSTRAGEM POR VIZINHANÇA (precisa de features/embedding) ---
#
# Diferente de class_balance_strategy e das duas acima, os candidatos aqui devem ser 4-tuplas
# (score, index, label, features) — features é o embedding da amostra (ex.: model.get_embedding(x)),
# necessário pra calcular vizinhança. Trata a classe com mais candidatos como "majoritária" e as
# demais, juntas, como "minoritária" (cenário mais comum na literatura destes métodos).

def _split_majority_minority(candidates):
    by_class = {}
    for c in candidates:
        by_class.setdefault(_class_key(c[2]), []).append(c)
    majority_key = max(by_class, key=lambda k: len(by_class[k]))
    majority = by_class.pop(majority_key)
    minority = [c for group in by_class.values() for c in group]
    return majority, minority


def tomek_links_strategy(candidates, **kwargs):
    """
    Tomek Links (Tomek, 1976, extensão do Condensed Nearest Neighbor de Hart, 1968): um par (a,b) de
    classes diferentes forma um Tomek Link se cada um é o vizinho mais próximo GLOBAL do outro — ou
    seja, não existe nenhum outro ponto (de qualquer classe) mais perto de a do que b, nem mais perto
    de b do que a. Remove o membro da classe MAJORITÁRIA de cada Tomek Link encontrado — limpa
    amostras majoritárias que "invadem" a fronteira de decisão da classe minoritária, sem alterar a
    minoritária.

    candidates: lista de (score, index, label, features) — precisa de features pra calcular
    vizinhança, diferente de class_balance_strategy.
    """
    if len(candidates) < 2:
        return list(candidates)

    majority, minority = _split_majority_minority(candidates)
    if not majority or not minority:
        return list(candidates)

    combined = majority + minority
    features = np.stack([c[3] for c in combined])
    is_majority = np.array([True] * len(majority) + [False] * len(minority))

    # Vizinho mais próximo GLOBAL de cada ponto (excluindo ele mesmo, por isso n_neighbors=2 e
    # descarta a coluna 0) — um único fit sobre TODOS os pontos, majoritário+minoritário juntos, é
    # o que a definição de Tomek Link exige (ver docstring). Ajustar o k-NN separadamente por classe
    # (majoritária vs. minoritária) over-identifica pares como Tomek Link: um ponto majoritário podia
    # ter um vizinho majoritário mais próximo ainda, que deveria desqualificar o par.
    nearest = NearestNeighbors(n_neighbors=2).fit(features).kneighbors(features, return_distance=False)[:, 1]

    tomek_positions = {
        i for i, j in enumerate(nearest)
        if is_majority[i] and not is_majority[j] and nearest[j] == i
    }

    print(f"Tomek Links: removendo {len(tomek_positions)} candidatos majoritarios de {len(majority)}.")
    kept_majority = [c for i, c in enumerate(majority) if i not in tomek_positions]
    return kept_majority + minority


def near_miss_strategy(candidates, target_count=None, version=1, n_neighbors=3):
    """
    NearMiss (Mani & Zhang, 2003, "kNN Approach to Unbalanced Data Distributions: A Case Study
    Involving Information Extraction"): reduz a classe majoritária mantendo as amostras mais
    "informativas" em relação à minoritária, em 3 variantes:

    - version=1: mantém as majoritárias com MENOR distância média aos n_neighbors vizinhos MAIS
      PRÓXIMOS da minoritária (perto da fronteira de decisão).
    - version=2: mantém as majoritárias com MENOR distância média aos n_neighbors vizinhos MAIS
      DISTANTES da minoritária (perto do "centro" da nuvem minoritária; evita outliers da fronteira).
    - version=3: algoritmo em duas etapas (Mani & Zhang, 2003). Etapa 1: para cada amostra
      minoritária, mantém seus n_neighbors vizinhos majoritários mais próximos (garante que toda
      região minoritária tenha vizinhança majoritária representada) — forma um conjunto candidato,
      tipicamente maior que target_count. Etapa 2: dentro desse candidato, mantém só os target_count
      com MAIOR distância média aos seus n_neighbors vizinhos minoritários mais próximos (descarta os
      mais "ruidosos"/redundantes do candidato).

    candidates: lista de (score, index, label, features). target_count: se não informado, iguala a
    contagem da minoritária — usado nas 3 versões (na v3, filtra o conjunto candidato da etapa 1).
    """
    majority, minority = _split_majority_minority(candidates)
    if not majority or not minority:
        return list(candidates)

    majority_features = np.stack([c[3] for c in majority])
    minority_features = np.stack([c[3] for c in minority])
    target_count = target_count or len(minority)

    if version == 3:
        k = min(n_neighbors, len(majority))
        idx = NearestNeighbors(n_neighbors=k).fit(majority_features).kneighbors(
            minority_features, return_distance=False)
        candidate_positions = sorted(set(idx.flatten().tolist()))

        # Etapa 2: dentro do candidato da etapa 1, mantém os target_count com MAIOR distância média
        # aos seus vizinhos minoritários mais próximos.
        candidate_features = majority_features[candidate_positions]
        k2 = min(n_neighbors, len(minority))
        distances, _ = NearestNeighbors(n_neighbors=k2).fit(minority_features).kneighbors(candidate_features)
        scores = distances.mean(axis=1)

        order = np.argsort(scores)[::-1]  # maior distância média primeiro
        kept_positions = [candidate_positions[i] for i in order[:target_count]]
        return [majority[i] for i in kept_positions] + minority

    if version == 1:
        k = min(n_neighbors, len(minority))
        distances, _ = NearestNeighbors(n_neighbors=k).fit(minority_features).kneighbors(majority_features)
        scores = distances.mean(axis=1)
    elif version == 2:
        k = min(n_neighbors, len(minority))
        distances, _ = NearestNeighbors(n_neighbors=len(minority)).fit(minority_features).kneighbors(majority_features)
        scores = distances[:, -k:].mean(axis=1)
    else:
        raise ValueError("near_miss_strategy: version deve ser 1, 2 ou 3.")

    order = np.argsort(scores)  # menor score primeiro, em todas as versões
    kept_majority = [majority[i] for i in order[:target_count]]
    return kept_majority + minority


# --- GERAÇÃO SINTÉTICA (precisa de features/embedding; devolve amostras SEM índice real) ---

def smote_oversample(candidates, target_count=None, k_neighbors=5, random_state=42):
    """
    SMOTE — Synthetic Minority Oversampling Technique (Chawla, Bowyer, Hall & Kegelmeyer, 2002):
    gera amostras SINTÉTICAS de classes minoritárias por interpolação linear entre uma amostra
    minoritária e um de seus k vizinhos mais próximos DA MESMA CLASSE, no espaço de features:

        x_new = x_i + λ·(x_j - x_i),  λ ~ U(0,1)

    candidates: lista de (score, index, label, features). Diferente das demais estratégias, NÃO
    seleciona entre candidatos existentes — devolve amostras sintéticas sem índice real no dataset
    (use com SyntheticEmbeddingDataset pra treinar com elas; exige que o modelo implemente
    classify_from_embedding(features), já que não há imagem por trás do embedding sintético).

    Devolve (candidates, synthetic): candidates é a lista original, sem alteração; synthetic é uma
    lista de (label, embedding_sintético).
    """
    by_class = {}
    for c in candidates:
        by_class.setdefault(_class_key(c[2]), []).append(c)

    target_count = target_count or max(len(group) for group in by_class.values())
    rng = np.random.default_rng(random_state)

    synthetic = []
    for group in by_class.values():
        n_needed = target_count - len(group)
        k = min(k_neighbors, len(group) - 1)
        if n_needed <= 0 or k < 1:
            continue  # nada a gerar, ou classe pequena demais pra ter vizinho da mesma classe

        features = np.stack([c[3] for c in group])
        original_label = group[0][2]
        # +1 vizinhos pq o mais próximo de cada ponto é ele mesmo (distância 0)
        neighbor_idx = NearestNeighbors(n_neighbors=k + 1).fit(features).kneighbors(features, return_distance=False)

        for _ in range(n_needed):
            base_pos = rng.integers(0, len(group))
            neighbor_pos = neighbor_idx[base_pos, rng.integers(1, k + 1)]
            lam = rng.uniform(0.0, 1.0)
            new_feat = features[base_pos] + lam * (features[neighbor_pos] - features[base_pos])
            synthetic.append((original_label, new_feat))

    return list(candidates), synthetic


def adasyn_oversample(candidates, target_count=None, k_neighbors=5, random_state=42):
    """
    ADASYN — Adaptive Synthetic Sampling (He, Bai, Garcia & Li, 2008, "ADASYN: Adaptive Synthetic
    Sampling Approach for Imbalanced Learning"): como o SMOTE, mas gera MAIS sintéticas para as
    instâncias minoritárias mais "difíceis" — perto da fronteira de decisão, com mais vizinhos de
    outra classe por perto. Para cada amostra minoritária i (entre os k vizinhos mais próximos,
    olhando TODOS os candidatos, não só a própria classe):

        r_i = (nº de vizinhos de outra classe) / k

    r_i é normalizado pra somar 1 dentro da classe, e as n_needed sintéticas totais são distribuídas
    proporcionalmente a r_i (amostras mais "na fronteira" recebem mais sintéticas) — cada sintética
    individual usa a mesma interpolação do SMOTE, entre a amostra e um vizinho DA MESMA CLASSE.

    Mesma interface de retorno que smote_oversample: (candidates, synthetic).
    """
    by_class = {}
    for c in candidates:
        by_class.setdefault(_class_key(c[2]), []).append(c)

    target_count = target_count or max(len(group) for group in by_class.values())
    rng = np.random.default_rng(random_state)

    all_features = np.stack([c[3] for c in candidates])
    all_labels = [_class_key(c[2]) for c in candidates]

    synthetic = []
    for label_key, group in by_class.items():
        n_needed = target_count - len(group)
        k_global = min(k_neighbors, len(all_features) - 1)
        k_local = min(k_neighbors, len(group) - 1)
        if n_needed <= 0 or k_global < 1 or k_local < 1:
            continue

        features = np.stack([c[3] for c in group])
        original_label = group[0][2]

        # r_i: fracao de vizinhos de OUTRA classe, entre os k_global vizinhos globais mais proximos
        global_idx = NearestNeighbors(n_neighbors=k_global + 1).fit(all_features).kneighbors(
            features, return_distance=False)
        r = np.array([
            sum(1 for j in row[1:] if all_labels[j] != label_key) / k_global
            for row in global_idx
        ])
        r = r / r.sum() if r.sum() > 0 else np.ones(len(group)) / len(group)
        n_synthetic_per_sample = np.round(r * n_needed).astype(int)

        local_idx = NearestNeighbors(n_neighbors=k_local + 1).fit(features).kneighbors(
            features, return_distance=False)

        for i, n_syn in enumerate(n_synthetic_per_sample):
            for _ in range(n_syn):
                neighbor_pos = local_idx[i, rng.integers(1, k_local + 1)]
                lam = rng.uniform(0.0, 1.0)
                new_feat = features[i] + lam * (features[neighbor_pos] - features[i])
                synthetic.append((original_label, new_feat))

    return list(candidates), synthetic


class SyntheticEmbeddingDataset(Dataset):
    """
    Dataset pra treinar com as amostras sintéticas do SMOTE/ADASYN — cada item já nasce como
    (embedding, label), sem imagem por trás. Combine com dados reais no mesmo DataLoader via
    ConcatDataset e use EmbeddingClassifierWrapper pra classificar a partir do embedding, pulando o
    backbone convolucional.
    """

    def __init__(self, synthetic_samples):
        self.samples = synthetic_samples  # lista de (label, embedding)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        label, embedding = self.samples[i]
        if not torch.is_tensor(embedding):
            embedding = torch.as_tensor(embedding, dtype=torch.float32)
        if not torch.is_tensor(label):
            label = torch.as_tensor(label)
        return embedding, label


class EmbeddingClassifierWrapper(nn.Module):
    """
    Envolve um modelo pra classificar direto a partir de um embedding pré-computado, pulando o
    backbone — necessário pra treinar com as amostras sintéticas de SMOTE/ADASYN (que não têm imagem
    por trás). Exige que o modelo implemente classify_from_embedding(features) (ver
    Models/models.py:SVHNCustomCNN).
    """

    def __init__(self, model):
        super().__init__()
        if not hasattr(model, "classify_from_embedding"):
            raise AttributeError("O modelo precisa implementar classify_from_embedding(features).")
        self.model = model

    def forward(self, embeddings):
        return self.model.classify_from_embedding(embeddings)
