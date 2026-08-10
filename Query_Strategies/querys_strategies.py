import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, kmeans_plusplus, AgglomerativeClustering

# Todas as estratégias recebem os mesmos kwargs (unlabeled_loader/unlabeled_indices e,
# quando aplicável, labeled_loader/labeled_indices/task) para que o ciclo de active learning
# possa chamá-las de forma uniforme, independente da estratégia escolhida.
#
# unlabeled_loader/labeled_loader devem ser DataLoaders com shuffle=False, iterando na
# mesma ordem de unlabeled_indices/labeled_indices, para que a posição de cada amostra no
# loader corresponda ao índice correspondente na lista de índices do dataset original.


def uncertainty_query_strategy(model, device, budget, unlabeled_loader, unlabeled_indices, task=None, **kwargs):
    """
    Seleciona as amostras mais incertas segundo task.compute_uncertainty (Training/tasks.py) — entropia
    para Classification/Segmentation, confiança média das detecções para Detection, margem de
    similaridade para VLM contrastivo.
    """
    if task is None:
        raise ValueError("uncertainty_query_strategy requer uma task (ver Training/tasks.py).")

    model.eval()
    all_uncertainties = []

    print(f"Calculando incerteza ({type(task).__name__}) para {len(unlabeled_indices)} amostras...")
    for batch in tqdm(unlabeled_loader):
        all_uncertainties.extend(task.compute_uncertainty(model, batch, device))

    all_uncertainties = np.array(all_uncertainties)

    # argsort ordena do menor para o maior, por isso pegamos os últimos do array
    selected_positions = np.argsort(all_uncertainties)[-budget:]

    return np.array(unlabeled_indices)[selected_positions]


def margin_query_strategy(model, device, budget, unlabeled_loader, unlabeled_indices, **kwargs):
    """
    Margin sampling (Scheffer, Decomain & Wrobel, 2001, "Active Hidden Markov Models for Information
    Extraction"): seleciona as amostras cuja diferença entre as probabilidades das duas classes mais
    prováveis é MENOR — o modelo está "em dúvida" entre seu 1º e 2º palpite.
    """
    model.eval()
    all_margins = []

    print(f"Calculando margem (top1 - top2) para {len(unlabeled_indices)} amostras...")
    with torch.no_grad():
        for batch in tqdm(unlabeled_loader):
            images = batch[0].to(device)
            probs = torch.softmax(model(images), dim=-1)  # [Batch, ..., num_classes]

            top2 = torch.topk(probs, k=min(2, probs.size(-1)), dim=-1).values
            margin = top2[..., 0] - top2[..., 1] if top2.size(-1) > 1 else top2[..., 0]

            # Reduz eventuais dimensões extras (ex.: várias posições/heads) para uma margem por amostra
            while margin.dim() > 1:
                margin = torch.mean(margin, dim=-1)

            all_margins.extend(margin.cpu().numpy())

    all_margins = np.array(all_margins)

    # Queremos as MENORES margens (mais ambíguas); argsort cresce, pegamos os primeiros
    selected_positions = np.argsort(all_margins)[:budget]
    return np.array(unlabeled_indices)[selected_positions]


def least_confidence_query_strategy(model, device, budget, unlabeled_loader, unlabeled_indices, **kwargs):
    """
    Least Confidence (Lewis & Gale, 1994, "A Sequential Algorithm for Training Text Classifiers"):
    seleciona as amostras com MENOR probabilidade máxima — o modelo está pouco confiante no seu
    próprio palpite mais provável. Equivalente à Variation Ratio (Freeman, 1965), usada como
    baseline determinística por Gal, Islam & Ghahramani (2017, "Deep Bayesian Active Learning with
    Image Data"): variation-ratio(x) = 1 - max_y p(y|x), a mesma fórmula usada aqui.
    """
    model.eval()
    all_scores = []

    print(f"Calculando least confidence (1 - max prob) para {len(unlabeled_indices)} amostras...")
    with torch.no_grad():
        for batch in tqdm(unlabeled_loader):
            images = batch[0].to(device)
            probs = torch.softmax(model(images), dim=-1)
            confidence = torch.max(probs, dim=-1).values  # [Batch, ...]

            while confidence.dim() > 1:
                confidence = torch.mean(confidence, dim=-1)

            all_scores.extend((1.0 - confidence).cpu().numpy())

    all_scores = np.array(all_scores)
    selected_positions = np.argsort(all_scores)[-budget:]
    return np.array(unlabeled_indices)[selected_positions]


# Variation Ratio (Freeman, 1965) é a mesma fórmula de Least Confidence para um único modelo
# determinístico — ver docstring acima e a seção 4 de Gal et al. (2017).
variation_ratio_query_strategy = least_confidence_query_strategy


def _enable_dropout(model):
    """Recoloca só as camadas de Dropout em modo treino, mantendo BatchNorm etc. em eval — o jeito
    correto de fazer MC-Dropout sem deixar BatchNorm recalcular estatísticas do batch de inferência."""
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def bald_query_strategy(model, device, budget, unlabeled_loader, unlabeled_indices, mc_samples=20, **kwargs):
    """
    BALD — Bayesian Active Learning by Disagreement (Houlsby, Huszár, Ghahramani & Lengyel, 2011),
    na aproximação prática via MC-Dropout de Gal, Islam & Ghahramani (2017, "Deep Bayesian Active
    Learning with Image Data"):

        I(x) ≈ H[ (1/T) Σ_t p_t(y|x) ] - (1/T) Σ_t H[p_t(y|x)]

    isto é, a entropia da predição MÉDIA entre T passes estocásticos (dropout ativo) menos a MÉDIA
    das entropias de cada passe individual. Mede incerteza epistêmica (desacordo entre os "modelos"
    amostrados via dropout) — diferente da entropia simples (task.compute_uncertainty), que mistura
    incerteza epistêmica com aleatória.

    Exige que o modelo tenha ao menos uma camada nn.Dropout; sem isso, os T passes seriam idênticos e
    o score colapsa a zero (BatchNorm é mantido em modo eval, usando as estatísticas já treinadas).

    mc_samples: T, número de passes estocásticos por amostra (padrão 20; o paper original usa T
    maior, ex. 100, para MNIST — mais T = estimativa mais estável, porém mais lenta).
    """
    model.eval()
    _enable_dropout(model)

    all_scores = []
    eps = 1e-10

    print(f"Calculando BALD (T={mc_samples} MC-Dropout) para {len(unlabeled_indices)} amostras...")
    with torch.no_grad():
        for batch in tqdm(unlabeled_loader):
            images = batch[0].to(device)

            pass_probs = torch.stack(
                [torch.softmax(model(images), dim=-1) for _ in range(mc_samples)], dim=0
            )  # [T, Batch, ..., C]

            mean_probs = pass_probs.mean(dim=0)  # [Batch, ..., C]
            entropy_of_mean = -torch.sum(mean_probs * torch.log(mean_probs + eps), dim=-1)  # [Batch, ...]

            per_pass_entropy = -torch.sum(pass_probs * torch.log(pass_probs + eps), dim=-1)  # [T, Batch, ...]
            mean_of_entropy = per_pass_entropy.mean(dim=0)  # [Batch, ...]

            bald_score = entropy_of_mean - mean_of_entropy
            while bald_score.dim() > 1:
                bald_score = torch.mean(bald_score, dim=-1)

            all_scores.extend(bald_score.cpu().numpy())

    all_scores = np.array(all_scores)
    selected_positions = np.argsort(all_scores)[-budget:]
    return np.array(unlabeled_indices)[selected_positions]


def extract_features(model, loader, device):
    """Extrai embeddings via model.get_embedding(x) — convenção exigida pelas estratégias abaixo."""
    if not hasattr(model, "get_embedding"):
        raise AttributeError(
            "O modelo precisa implementar get_embedding(x) para usar estratégias baseadas em embedding."
        )

    model.eval()
    features_list = []

    print(f"Extraindo embeddings para {len(loader.dataset)} amostras...")
    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch[0].to(device)
            feat = model.get_embedding(images)
            features_list.append(feat.cpu().numpy())

    return np.concatenate(features_list, axis=0)


# --- ESTRATÉGIA: DENSIDADE (K-MEANS CENTERS) ---
def density_query_strategy(model, device, budget, unlabeled_loader, unlabeled_indices, **kwargs):
    """Seleciona amostras representativas dos centros de clusters."""
    features = extract_features(model, unlabeled_loader, device)

    # Criamos K clusters (onde K = budget)
    print(f"Agrupando em {budget} clusters...")
    kmeans = KMeans(n_clusters=budget, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    centers = kmeans.cluster_centers_

    # Para cada centro, pegamos a amostra real mais próxima a ele
    selected_indices = []
    for i in range(budget):
        cluster_points_idx = np.where(cluster_labels == i)[0]
        distances = cdist(centers[i:i + 1], features[cluster_points_idx], 'euclidean')
        closest_point_in_cluster = cluster_points_idx[np.argmin(distances)]
        selected_indices.append(unlabeled_indices[closest_point_in_cluster])

    return np.array(selected_indices)


# --- ESTRATÉGIA: DIVERSIDADE (K-CENTER GREEDY / DISTANCE) ---
def diversity_query_strategy(model, device, budget, unlabeled_loader, unlabeled_indices,
                              labeled_loader=None, labeled_indices=None, **kwargs):
    """Seleciona amostras mais distantes do conjunto já rotulado."""
    if labeled_loader is None or labeled_indices is None:
        raise ValueError("diversity_query_strategy requer labeled_loader e labeled_indices.")

    feat_unlabeled = extract_features(model, unlabeled_loader, device)
    feat_labeled = extract_features(model, labeled_loader, device)

    print("Calculando distâncias para diversidade...")
    # Distância mínima de cada ponto não rotulado para o "vizinho rotulado mais próximo"
    min_distances = np.min(cdist(feat_unlabeled, feat_labeled, 'euclidean'), axis=1)

    selected_indices = []
    for _ in range(budget):
        # Seleciona o ponto que tem a MAIOR "distância mínima" (o mais isolado)
        idx = np.argmax(min_distances)
        selected_indices.append(unlabeled_indices[idx])

        # O novo ponto agora é "rotulado": novos cálculos devem considerar a distância para ele também
        new_feat = feat_unlabeled[idx:idx + 1]
        dist_to_new = cdist(feat_unlabeled, new_feat, 'euclidean').flatten()
        min_distances = np.minimum(min_distances, dist_to_new)

    return np.array(selected_indices)


# --- ESTRATÉGIA: BADGE (GRADIENTE + DIVERSIDADE) ---
def badge_query_strategy(model, device, budget, unlabeled_loader, unlabeled_indices, **kwargs):
    """
    BADGE — Batch Active learning by Diverse Gradient Embeddings (Ash, Zhang, Krishnamurthy, Langford
    & Agarwal, 2020, ICLR, "Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds"):
    combina incerteza e diversidade sem hiperparâmetro extra.

    Para cada amostra x, computa o "gradiente hipotético" da última camada, usando a própria predição
    do modelo ŷ = argmax p(y|x) como rótulo hipotético (Eq. 1 do paper):

        g_x[i] = (p_i - 1{i=ŷ}) ⊗ z(x)

    onde z(x) = model.get_embedding(x) é a penúltima camada e p é o softmax. O paper mostra que
    ||g_x|| é uma cota inferior para o gradiente induzido pelo rótulo verdadeiro — pequena quando o
    modelo já está confiante (pouco a aprender ali), grande quando está incerto — então o vetor já
    embute incerteza por construção. A seleção do batch usa o algoritmo de SEEDING do k-MEANS++
    (Arthur & Vassilvitskii, 2007) sobre {g_x}, que favorece pontos de alta magnitude (incertos) e
    espalhados entre si (diversos), sem rodar k-means completo nem introduzir hiperparâmetros novos.
    """
    if not hasattr(model, "get_embedding"):
        raise AttributeError("O modelo precisa implementar get_embedding(x) para usar BADGE.")

    model.eval()
    gradient_embeddings = []

    print(f"Calculando embeddings de gradiente (BADGE) para {len(unlabeled_indices)} amostras...")
    with torch.no_grad():
        for batch in tqdm(unlabeled_loader):
            images = batch[0].to(device)
            features = model.get_embedding(images)  # [Batch, D]
            probs = torch.softmax(model(images), dim=-1)  # [Batch, ..., C]
            preds = torch.argmax(probs, dim=-1)  # [Batch, ...]
            one_hot = F.one_hot(preds, num_classes=probs.size(-1)).float()  # [Batch, ..., C]

            diff = probs - one_hot
            if diff.dim() > 2:
                diff = diff.reshape(diff.size(0), -1)  # [Batch, C'] (achata posições extras, se houver)

            grad_embedding = diff.unsqueeze(-1) * features.unsqueeze(1)  # [Batch, C', D]
            grad_embedding = grad_embedding.reshape(grad_embedding.size(0), -1)  # [Batch, C'*D]

            gradient_embeddings.append(grad_embedding.cpu().numpy())

    gradient_embeddings = np.concatenate(gradient_embeddings, axis=0)

    print(f"Selecionando {budget} amostras via k-means++ seeding sobre os embeddings de gradiente...")
    _, selected_positions = kmeans_plusplus(gradient_embeddings, n_clusters=budget, random_state=42)

    return np.array(unlabeled_indices)[selected_positions]


def _round_robin_select(sorted_clusters, budget, rng):
    """Amostragem round-robin (Algorithm 2 do paper Cluster-Margin): embaralha cada cluster, depois
    percorre os clusters em ordem cíclica pegando um exemplo por vez, pulando os já esgotados."""
    for cluster in sorted_clusters:
        rng.shuffle(cluster)

    cursors = [0] * len(sorted_clusters)
    selected = []
    j = 0
    while len(selected) < budget:
        attempts = 0
        while cursors[j] >= len(sorted_clusters[j]):
            j = (j + 1) % len(sorted_clusters)
            attempts += 1
            if attempts > len(sorted_clusters):
                return selected  # todos os clusters candidatos já esgotados
        selected.append(sorted_clusters[j][cursors[j]])
        cursors[j] += 1
        j = (j + 1) % len(sorted_clusters)
    return selected


# --- ESTRATÉGIA: CLUSTER-MARGIN (MARGIN + DIVERSIDADE VIA CLUSTERING ÚNICO) ---
class ClusterMarginQueryStrategy:
    """
    Cluster-Margin (Citovsky, DeSalvo, Gentile, Karydas, Rajagopalan, Rostamizadeh & Kumar, 2021,
    NeurIPS, "Batch Active Learning at Scale"): como o Margin, mas diversifica o batch escolhido via
    clustering — evita mandar pro oráculo um monte de amostras redundantes com margem baixa pelo
    mesmo motivo.

    É uma classe (não uma função) porque o paper roda o clustering (HAC) só UMA VEZ, como
    pré-processamento, e reaproveita esse resultado em todas as chamadas seguintes — reclusterizar a
    cada ciclo custaria O(n² log n) sempre que chamado. self._cluster_labels fica em cache após a
    primeira chamada.

    Algorithm 1 (uma vez, na primeira chamada): Hierarchical Agglomerative Clustering (HAC) com
    average-linkage sobre os embeddings do pool não rotulado inteiro.

    Algorithm 2 (a cada chamada): pega os k_m candidatos de MENOR margem (k_m = margin_multiplier ×
    budget), mapeia pros clusters do HAC, ordena os clusters tocados por tamanho (crescente) e faz
    amostragem round-robin — um exemplo aleatório por cluster, começando pelos menores — até
    completar o budget. Clusters pequenos/raros entram primeiro, e o round-robin espalha a escolha
    por muitos clusters em vez de concentrar tudo num cluster popular.
    """

    def __init__(self, margin_multiplier=10, hac_distance_threshold=None, n_clusters=None, random_state=42):
        """
        margin_multiplier: k_m / budget — quantos candidatos de menor margem considerar antes de
        diversificar (padrão 10x).
        hac_distance_threshold / n_clusters: controla onde o HAC para de agrupar; se nenhum for
        informado, usa n_clusters = sqrt(tamanho do pool) como heurística.
        """
        self.margin_multiplier = margin_multiplier
        self.hac_distance_threshold = hac_distance_threshold
        self.n_clusters = n_clusters
        self._cluster_labels = None
        self._clustered_indices = None
        self._rng = np.random.default_rng(random_state)

    def _fit_clusters(self, model, device, unlabeled_loader, unlabeled_indices):
        print("Cluster-Margin: rodando HAC (uma única vez) sobre os embeddings do pool não rotulado...")
        features = extract_features(model, unlabeled_loader, device)

        n_clusters = self.n_clusters or max(2, int(np.sqrt(len(features))))
        clustering = AgglomerativeClustering(
            n_clusters=None if self.hac_distance_threshold else n_clusters,
            distance_threshold=self.hac_distance_threshold,
            linkage="average",
        )
        self._cluster_labels = clustering.fit_predict(features)
        self._clustered_indices = np.array(unlabeled_indices)
        print(f"Cluster-Margin: {len(np.unique(self._cluster_labels))} clusters encontrados.")

    def __call__(self, model, device, budget, unlabeled_loader, unlabeled_indices, **kwargs):
        if self._cluster_labels is None:
            self._fit_clusters(model, device, unlabeled_loader, unlabeled_indices)
        index_to_cluster = dict(zip(self._clustered_indices.tolist(), self._cluster_labels.tolist()))

        model.eval()
        all_margins = []
        print(f"Calculando margem para {len(unlabeled_indices)} amostras...")
        with torch.no_grad():
            for batch in tqdm(unlabeled_loader):
                images = batch[0].to(device)
                probs = torch.softmax(model(images), dim=-1)
                top2 = torch.topk(probs, k=min(2, probs.size(-1)), dim=-1).values
                margin = top2[..., 0] - top2[..., 1] if top2.size(-1) > 1 else top2[..., 0]
                while margin.dim() > 1:
                    margin = torch.mean(margin, dim=-1)
                all_margins.extend(margin.cpu().numpy())

        all_margins = np.array(all_margins)
        unlabeled_indices = np.array(unlabeled_indices)

        k_m = min(len(unlabeled_indices), self.margin_multiplier * budget)
        candidates = unlabeled_indices[np.argsort(all_margins)[:k_m]]

        clusters_to_candidates = {}
        for idx in candidates:
            cluster_id = index_to_cluster.get(int(idx), -1)
            clusters_to_candidates.setdefault(cluster_id, []).append(int(idx))
        sorted_clusters = sorted(clusters_to_candidates.values(), key=len)

        selected = _round_robin_select(sorted_clusters, min(budget, len(candidates)), self._rng)
        return np.array(selected)
