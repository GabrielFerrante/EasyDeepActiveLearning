import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# LPL — Learning Loss for Active Learning (Yoo & Kweon, 2019, CVPR, "Learning Loss for Active
# Learning"): diferente das outras Query_Strategies, não é uma função pura sobre o modelo alvo — o
# LossPredictionModule precisa ser criado, treinado JUNTO com o modelo alvo (via train_model_with_lpl,
# que substitui Training/train.py só pros ciclos que usam LPL) e então usado na hora de consultar.
#
# Convenção exigida do modelo alvo: model.get_intermediate_features(x) -> lista de mapas de ativação
# [Batch, C_i, H_i, W_i] de diferentes estágios da rede (ver Models/models.py, SVHNCustomCNN).


class LossPredictionModule(nn.Module):
    """
    Módulo de previsão de loss (Fig. 2 do paper): conectado a features intermediárias do modelo alvo,
    aprende a prever a loss que o modelo alvo teria numa amostra, sem precisar do rótulo dela.

    Para cada mapa de features h_i: Global Average Pooling -> FC -> ReLU, produzindo um vetor de
    tamanho hidden_dim por estágio. Os vetores de todos os estágios são concatenados e passam por uma
    última FC, produzindo um escalar (a loss prevista).

    feature_channels: lista com o número de canais de cada mapa de features que
    model.get_intermediate_features(x) devolve, na mesma ordem (ex.: [128, 256, 512, 512] pro
    SVHNCustomCNN, que usa layer3..layer6).
    """

    def __init__(self, feature_channels, hidden_dim=128):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fcs = nn.ModuleList([nn.Linear(c, hidden_dim) for c in feature_channels])
        self.relu = nn.ReLU()
        self.final_fc = nn.Linear(hidden_dim * len(feature_channels), 1)

    def forward(self, feature_maps):
        pooled = []
        for feat, fc in zip(feature_maps, self.fcs):
            v = self.gap(feat).flatten(1)  # [Batch, C_i]
            pooled.append(self.relu(fc(v)))  # [Batch, hidden_dim]
        combined = torch.cat(pooled, dim=1)  # [Batch, hidden_dim * n_estágios]
        return self.final_fc(combined).squeeze(-1)  # [Batch]


def loss_prediction_loss(predicted_loss, target_loss, margin=1.0):
    """
    Loss do LPL (Eq. 2 do paper): compara PARES de amostras dentro do batch em vez de regredir a
    loss real diretamente (MSE) — o paper mostra que MSE não funciona bem porque a escala da loss
    real muda com o tempo (decresce conforme o modelo alvo aprende), e regredir pra um alvo que muda
    de escala prejudica o módulo. Em vez disso, o módulo só precisa aprender a ORDEM relativa:

        L(l̂,l) = max(0, -sign(l_i - l_j)·(l̂_i - l̂_j) + ξ)

    O batch é dividido em pares consecutivos (2i, 2i+1); se a amostra i tem loss real maior que a
    amostra j, o módulo é penalizado a menos que também preveja l̂_i > l̂_j com margem ξ.
    """
    batch_size = predicted_loss.size(0)
    if batch_size % 2 != 0:
        predicted_loss = predicted_loss[:-1]
        target_loss = target_loss[:-1]
        batch_size -= 1
    if batch_size == 0:
        return predicted_loss.sum() * 0.0

    pred_i, pred_j = predicted_loss[0::2], predicted_loss[1::2]
    true_i, true_j = target_loss[0::2], target_loss[1::2]

    sign = torch.sign(true_i - true_j)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)  # empate: trata como +1, por convenção

    return torch.clamp(-sign * (pred_i - pred_j) + margin, min=0.0).mean()


def train_model_with_lpl(
    model, loss_prediction_module, train_loader, test_loader, task, per_sample_loss_fn,
    optimizer, lpl_optimizer, device, epochs,
    lambda_lpl=1.0, margin=1.0, detach_after_fraction=0.6, writer=None, log_prefix="",
):
    """
    Variante de Training/train.py:train_model que também treina o LossPredictionModule junto com o
    modelo alvo (Eq. 1 do paper: L_target + λ·L_loss).

    per_sample_loss_fn(outputs, targets) -> Tensor[Batch]: a MESMA loss de task, mas sem reduzir a
    escalar — o LPL precisa da loss de cada amostra individualmente pra aprender a ordenar. Ex., pra
    uma ClassificationTask: lambda outputs, targets: F.cross_entropy(outputs, targets, reduction="none").

    detach_after_fraction: fração do total de épocas após a qual a loss do LPL para de propagar
    gradiente pro modelo alvo (só o módulo LPL continua aprendendo com as features já fixas) — evita
    que a tarefa auxiliar atrapalhe as features do modelo principal perto do fim do treino (mesmo
    truque do paper original, lá aplicado após 120 de 200 épocas ≈ 0.6).
    """
    loss = 0.0
    test_metric = 0.0
    for epoch in range(epochs):
        model.train()
        loss_prediction_module.train()
        detach_features = (epoch / epochs) >= detach_after_fraction

        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Train+LPL [{epoch + 1}/{epochs}]", unit="batch", leave=False)
        for batch in progress_bar:
            inputs, targets = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            lpl_optimizer.zero_grad()

            outputs = model(inputs)
            per_sample = per_sample_loss_fn(outputs, targets)  # [Batch]
            target_loss = per_sample.mean()

            feature_maps = model.get_intermediate_features(inputs)
            if detach_features:
                feature_maps = [f.detach() for f in feature_maps]
            predicted_loss = loss_prediction_module(feature_maps)

            lpl_loss = loss_prediction_loss(predicted_loss, per_sample.detach(), margin=margin)
            total_loss = target_loss + lambda_lpl * lpl_loss
            total_loss.backward()

            optimizer.step()
            lpl_optimizer.step()

            running_loss += target_loss.item()
            progress_bar.set_postfix(loss=f"{target_loss.item():.4f}", lpl_loss=f"{lpl_loss.item():.4f}")

        loss = running_loss / len(train_loader)
        test_metric = _evaluate(model, test_loader, task, device, epoch, epochs)
        print(f"Epoch {epoch + 1}/{epochs}: Loss {loss:.4f} | Test Metric: {test_metric:.4f}")

        if writer is not None:
            writer.add_scalar(f"{log_prefix}Loss/train", loss, epoch)
            writer.add_scalar(f"{log_prefix}Metric/test", test_metric, epoch)

    return loss, test_metric


def _evaluate(model, loader, task, device, epoch, epochs):
    model.eval()
    all_metrics = []
    progress_bar = tqdm(loader, desc=f"Eval [{epoch + 1}/{epochs}]", unit="batch", leave=False)
    for batch in progress_bar:
        all_metrics.extend(task.compute_metric(model, batch, device))
    return float(np.mean(all_metrics))


def lpl_query_strategy(model, loss_prediction_module, device, budget, unlabeled_loader, unlabeled_indices, **kwargs):
    """
    Seleciona as amostras com MAIOR loss prevista pelo LossPredictionModule já treinado — as que o
    modelo alvo provavelmente erraria mais.

    Diferente das outras Query_Strategies, precisa do loss_prediction_module treinado, que não faz
    parte dos kwargs padrão passados por run_active_learning_cycle/run_self_labeling_cycle. Para
    plugar aqui, amarre o módulo treinado antes de passar como query_strategy, ex.:

        query_strategy = lambda **kw: lpl_query_strategy(loss_prediction_module=trained_module, **kw)
    """
    model.eval()
    loss_prediction_module.eval()
    all_scores = []

    print(f"Calculando loss prevista (LPL) para {len(unlabeled_indices)} amostras...")
    with torch.no_grad():
        for batch in tqdm(unlabeled_loader):
            images = batch[0].to(device)
            feature_maps = model.get_intermediate_features(images)
            predicted_loss = loss_prediction_module(feature_maps)
            all_scores.extend(predicted_loss.cpu().numpy())

    all_scores = np.array(all_scores)
    selected_positions = np.argsort(all_scores)[-budget:]
    return np.array(unlabeled_indices)[selected_positions]
