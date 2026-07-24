import torch
import numpy as np
from torch.utils.data import Dataset

# Diferente das Query_Strategies (que escolhem QUAIS amostras mandar pro oráculo), as estratégias
# daqui decidem COMO obter o rótulo de uma amostra não rotulada. A pseudo-labeling clássica não
# envolve oráculo nenhum: usa a própria predição do modelo como rótulo, quando ele está confiante o
# suficiente.


def pseudo_labeling_strategy(model, device, unlabeled_loader, unlabeled_indices, balance_fn,
                              confidence_threshold=0.95, max_per_class=None, **kwargs):
    """
    Pseudo-labeling clássico (Lee, 2013): roda o modelo sobre o pool não rotulado e junta como
    candidato a pseudo-rótulo a predição (argmax do softmax) de toda amostra cuja confiança
    (probabilidade máxima) seja >= confidence_threshold. O balanceamento por classe entre os
    candidatos aprovados é feito por balance_fn (ver Balance_Strategies/balance_strategies.py) — sem
    isso, a classe que o modelo já acerta mais (logo, mais confiante) tende a dominar o conjunto
    pseudo-rotulado e reforçar esse viés a cada ciclo.

    balance_fn: callable(candidates, max_per_class) -> candidatos balanceados, onde candidates é uma
    lista de (confidence, dataset_index, pred). Ex.: Balance_Strategies.balance_strategies.class_balance_strategy.

    unlabeled_loader deve ser um DataLoader com shuffle=False, iterando na mesma ordem de
    unlabeled_indices (ver Query_Strategies/querys_strategies.py).

    Devolve (selected_indices, pseudo_labels): os índices originais do dataset aceitos e os rótulos
    previstos correspondentes (na mesma ordem), já balanceados por classe.
    """
    model.eval()
    candidates = []  # [(confidence, dataset_index, pred_tensor), ...]

    position = 0
    with torch.no_grad():
        for batch in unlabeled_loader:
            images = batch[0].to(device)
            outputs = model(images)  # [Batch, ..., num_classes]

            probs = torch.softmax(outputs, dim=-1)
            per_position_confidence, preds = torch.max(probs, dim=-1)

            # Reduz eventuais dimensões extras (ex.: várias posições/heads) para uma confiança por amostra
            sample_confidence = per_position_confidence
            while sample_confidence.dim() > 1:
                sample_confidence = torch.mean(sample_confidence, dim=-1)

            batch_size = images.size(0)
            for i in range(batch_size):
                confidence = sample_confidence[i].item()
                if confidence >= confidence_threshold:
                    candidates.append((confidence, unlabeled_indices[position + i], preds[i].cpu()))
            position += batch_size

    balanced = balance_fn(candidates, max_per_class=max_per_class)

    selected_indices = [index for _, index, _ in balanced]
    pseudo_labels = [pred for _, _, pred in balanced]
    return np.array(selected_indices, dtype=int), pseudo_labels


class PseudoLabeledDataset(Dataset):
    """
    Dataset dedicado a amostras nunca rotuladas: para cada índice, pega a imagem do dataset base e
    devolve o pseudo-rótulo previsto pelo modelo no lugar do rótulo — o rótulo real do dataset base
    é descartado sem nunca ser usado. Mantém os eventuais elementos extras do item original (ex.:
    o "length" do SVHNCustomDataset) pra ter a mesma aridade do conjunto rotulado de verdade, já que
    os dois são combinados num único DataLoader via ConcatDataset (misturar tuplas de tamanhos
    diferentes no mesmo batch quebraria o collate do PyTorch). Pseudo-rotulação nunca sobrescreve ou
    observa o rótulo de uma amostra já rotulada por um oráculo.
    """

    def __init__(self, base_dataset, indices, pseudo_labels):
        self.base_dataset = base_dataset
        self.indices = indices
        self.pseudo_labels = pseudo_labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        item = self.base_dataset[self.indices[i]]
        return (item[0], self.pseudo_labels[i]) + tuple(item[2:])