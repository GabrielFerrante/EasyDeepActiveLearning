import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

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
