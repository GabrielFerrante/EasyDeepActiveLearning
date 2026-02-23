import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

def entropy_query_strategy(model, dataset, unlabeled_indices, budget, batch_size, device):
    """
    Seleciona as amostras com maior entropia (maior incerteza).
    """
    model.eval()
    all_entropies = []
    
    # Criamos um loader para o pool não rotulado
    unlabeled_subset = Subset(dataset, unlabeled_indices)
    loader = DataLoader(unlabeled_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Calculando entropia para {len(unlabeled_indices)} amostras...")
    
    with torch.no_grad():
        for images, _, _ in tqdm(loader):
            images = images.to(device)
            outputs = model(images)  # Shape: [Batch, 5, 11]
            
            
            probs = torch.softmax(outputs, dim=2) # [Batch, 5, 11]
            
            # -sum(p * log(p))
            #  epsilon (1e-10) para evitar log(0)
            log_probs = torch.log(probs + 1e-10)
            entropy_per_digit = -torch.sum(probs * log_probs, dim=2) # [Batch, 5]
            
            # Média da entropia das 5 posições (incerteza global da imagem)
            mean_entropy = torch.mean(entropy_per_digit, dim=1) # [Batch]
            
            all_entropies.extend(mean_entropy.cpu().numpy())

    all_entropies = np.array(all_entropies)
    
    #  Selecionar os índices com MAIOR entropia
    # argsort ordena do menor para o maior, por isso pegamos os últimos do array
    selected_subset_indices = np.argsort(all_entropies)[-budget:]
    
    # Retornamos os índices originais do dataset
    return np.array(unlabeled_indices)[selected_subset_indices]

def extract_features(model, dataset, indices, batch_size, device):
    """Extrai o vetor de características da última camada antes das 'heads'."""
    model.eval()
    features_list = []
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    print(f" Extraindo características para {len(indices)} amostras...")
    with torch.no_grad():
        for images, _, _ in loader:
            images = images.to(device)
            # Passamos pela base da CNN (backbone)
            # Na sua classe SVHNCustomCNN, o forward passa pelas camadas e faz avgpool
            x = model.layer1(images)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.layer5(x)
            x = model.layer6(x)
            x = model.avgpool(x)
            feat = torch.flatten(x, 1)
            features_list.append(feat.cpu().numpy())
            
    return np.concatenate(features_list, axis=0)

# --- ESTRATÉGIA 1: DENSIDADE (K-MEANS CENTERS) ---
def density_query_strategy(model, dataset, unlabeled_indices, budget, batch_size, device):
    """Seleciona amostras representativas dos centros de clusters."""
    features = extract_features(model, dataset, unlabeled_indices, batch_size, device)
    
    # Criamos K clusters (onde K = budget)
    print(f" Agrupando em {budget} clusters...")
    kmeans = KMeans(n_clusters=budget, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    centers = kmeans.cluster_centers_

    # Para cada centro, pegamos a amostra real mais próxima a ele
    selected_indices = []
    for i in range(budget):
        cluster_points_idx = np.where(cluster_labels == i)[0]
        distances = cdist(centers[i:i+1], features[cluster_points_idx], 'euclidean')
        closest_point_in_cluster = cluster_points_idx[np.argmin(distances)]
        selected_indices.append(unlabeled_indices[closest_point_in_cluster])
        
    return np.array(selected_indices)

# --- ESTRATÉGIA 2: DIVERSIDADE (K-CENTER GREEDY / DISTANCE) ---
def diversity_query_strategy(model, dataset, labeled_indices, unlabeled_indices, budget, batch_size, device):
    """Seleciona amostras mais distantes do conjunto já rotulado."""
    # Extraímos features de ambos
    feat_unlabeled = extract_features(model, dataset, unlabeled_indices, batch_size, device)
    feat_labeled = extract_features(model, dataset, labeled_indices, batch_size, device)

    print(f" Calculando distâncias para diversidade...")
    # Calcula distância mínima de cada ponto não rotulado para qualquer ponto rotulado
    # Usamos o mínimo das distâncias para encontrar o "vizinho rotulado mais próximo"
    min_distances = np.min(cdist(feat_unlabeled, feat_labeled, 'euclidean'), axis=1)

    selected_indices = []
    for _ in range(budget):
        # Seleciona o ponto que tem a MAIOR "distância mínima" (o mais isolado)
        idx = np.argmax(min_distances)
        selected_indices.append(unlabeled_indices[idx])
        
        # Atualiza distâncias: o novo ponto agora é "rotulado", 
        # novos cálculos devem considerar a distância para ele também
        new_feat = feat_unlabeled[idx:idx+1]
        dist_to_new = cdist(feat_unlabeled, new_feat, 'euclidean').flatten()
        min_distances = np.minimum(min_distances, dist_to_new)

    return np.array(selected_indices)