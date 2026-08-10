import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# WAAL — Wasserstein Adversarial Active Learning (Shui, Zhou, Gagné & Wang, 2020, AISTATS, "Deep
# Active Learning: Unified and Principled Method for Query and Training"). Diferente de VAAL, não usa
# um VAE à parte: o crítico opera direto sobre model.get_embedding(x), o mesmo espaço de features do
# modelo alvo — e o próprio treino do modelo alvo é modificado (loss conjunta de tarefa + termo
# adversarial), não só a estratégia de seleção.


class WassersteinCritic(nn.Module):
    """Crítico 1-Lipschitz (imposto via gradient penalty, estilo WGAN-GP) sobre o espaço de
    embeddings do modelo alvo — estima a distância de Wasserstein entre a distribuição rotulada e a
    não rotulada nesse espaço."""

    def __init__(self, embedding_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


def _gradient_penalty(critic, z_labeled, z_unlabeled, device):
    """Penalidade de gradiente (Gulrajani et al., 2017) pra manter o crítico 1-Lipschitz: interpola
    entre embeddings rotulados e não rotulados e penaliza a norma do gradiente do crítico se afastar
    de 1 nesses pontos interpolados."""
    batch_size = min(z_labeled.size(0), z_unlabeled.size(0))
    z_l, z_u = z_labeled[:batch_size], z_unlabeled[:batch_size]

    eps = torch.rand(batch_size, 1, device=device)
    z_hat = (eps * z_l + (1 - eps) * z_u).clone().requires_grad_(True)
    scores = critic(z_hat)
    grads = torch.autograd.grad(
        outputs=scores, inputs=z_hat, grad_outputs=torch.ones_like(scores),
        create_graph=True, retain_graph=True,
    )[0]
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


def bias_coefficient(num_labeled, num_unlabeled, budget):
    """
    C0 = (γ-α) / (γ(1+α)), com γ=U/L (razão de desbalanço) e α=B/L (razão da query) — Eq. 8 do
    paper. Corrige o termo adversarial pelo desbalanço entre o tamanho do pool rotulado e o não
    rotulado (a "redundancy trick": como L << U, o termo adversarial cru favoreceria demais o não
    rotulado sem essa correção).
    """
    gamma = num_unlabeled / max(num_labeled, 1)
    alpha = budget / max(num_labeled, 1)
    return (gamma - alpha) / (gamma * (1 + alpha))


def _evaluate(model, loader, task, device, epoch, epochs):
    model.eval()
    all_metrics = []
    progress_bar = tqdm(loader, desc=f"Eval [{epoch + 1}/{epochs}]", unit="batch", leave=False)
    for batch in progress_bar:
        all_metrics.extend(task.compute_metric(model, batch, device))
    return float(np.mean(all_metrics))


def train_model_with_waal(
    model, critic, train_loader, unlabeled_loader, test_loader, task,
    optimizer, critic_optimizer, device, epochs,
    num_labeled, num_unlabeled, budget,
    mu=1.0, gp_weight=10.0, critic_steps=5, writer=None, log_prefix="",
):
    """
    Algorithm 1 do paper ("DNN Parameter Training Stage"). A cada mini-batch rotulado (train_loader,
    shuffle=True), pareia com um mini-batch não rotulado de mesmo tamanho (reciclando o loader menor
    — a "redundancy trick" do paper). Duas atualizações por passo:

    1. Crítico: treinado por `critic_steps` iterações (estilo WGAN) pra maximizar a estimativa de
       Wasserstein entre embeddings rotulados/não-rotulados, sujeito à penalidade de gradiente.
    2. Modelo alvo (feature extractor + classificador, atualizados juntos num único backward): loss
       de tarefa (task.criterion) + μ vezes o mesmo termo adversarial — como o crítico só enxerga
       embeddings (não os parâmetros do classificador), o gradiente do termo adversarial só afeta o
       extrator de features, exatamente como o Algorithm 1 descreve em separado.

    Requer task com atributo .criterion (ex.: ClassificationTask, SegmentationTask) — WAAL foi
    formulado pro caso de classificação com softmax (as cotas de incerteza da query stage assumem
    isso).
    """
    if not hasattr(model, "get_embedding"):
        raise AttributeError("O modelo precisa implementar get_embedding(x) para usar WAAL.")
    if not hasattr(task, "criterion"):
        raise AttributeError("WAAL exige uma task com atributo .criterion (ex.: ClassificationTask).")

    c0 = bias_coefficient(num_labeled, num_unlabeled, budget)
    loss = 0.0
    test_metric = 0.0

    for epoch in range(epochs):
        model.train()
        critic.train()

        unlabeled_iter = iter(unlabeled_loader)
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Train+WAAL [{epoch + 1}/{epochs}]", unit="batch", leave=False)
        for labeled_batch in progress_bar:
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)

            x_l, y_l = labeled_batch[0].to(device), labeled_batch[1].to(device)
            x_u = unlabeled_batch[0].to(device)

            # --- 1) Crítico (algumas iterações, embeddings sem gradiente pro extrator) ---
            with torch.no_grad():
                z_l_det = model.get_embedding(x_l)
                z_u_det = model.get_embedding(x_u)

            for _ in range(critic_steps):
                critic_optimizer.zero_grad()
                critic_term = critic(z_u_det).mean() - c0 * critic(z_l_det).mean()
                gp = _gradient_penalty(critic, z_l_det, z_u_det, device)
                critic_loss = -critic_term + gp_weight * gp  # ascender em critic_term = descer em -critic_term
                critic_loss.backward()
                critic_optimizer.step()

            # --- 2) Modelo alvo (θ^f e θ^h juntos, um único backward) ---
            optimizer.zero_grad()
            outputs = model(x_l)
            task_loss = task.criterion(outputs, y_l)

            z_l = model.get_embedding(x_l)
            z_u = model.get_embedding(x_u)
            critic_term = critic(z_u).mean() - c0 * critic(z_l).mean()

            total_loss = task_loss + mu * critic_term
            total_loss.backward()
            optimizer.step()

            running_loss += task_loss.item()
            progress_bar.set_postfix(task_loss=f"{task_loss.item():.4f}", critic=f"{critic_term.item():.4f}")

        loss = running_loss / len(train_loader)
        test_metric = _evaluate(model, test_loader, task, device, epoch, epochs)
        print(f"Epoch {epoch + 1}/{epochs}: Loss {loss:.4f} | Test Metric: {test_metric:.4f}")

        if writer is not None:
            writer.add_scalar(f"{log_prefix}Loss/train", loss, epoch)
            writer.add_scalar(f"{log_prefix}Metric/test", test_metric, epoch)

    return loss, test_metric


def waal_query_strategy(model, critic, device, budget, unlabeled_loader, unlabeled_indices,
                         mu=1.0, uncertainty_weight=0.5, **kwargs):
    """
    Querying Stage do paper (Eq. 9-11): seleciona as B amostras que MINIMIZAM U(x) - μ·g(x), onde:

    - U(x) é uma combinação convexa de duas cotas superiores de incerteza — Eq. 10 (pior caso: o
      rótulo hipotético é a classe menos provável, -log(min_y p(y|x))) e Eq. 11 (soma sobre todas as
      classes, -Σ_y log p(y|x)). Ambas ficam MENORES quanto mais uniforme (incerta) for a predição.
    - g(x) é o crítico de Wasserstein já treinado (train_model_with_waal) sobre model.get_embedding —
      maior quando a amostra "parece" mais distante da distribuição rotulada atual (mais informativa
      pra diversidade).

    uncertainty_weight pondera a combinação convexa entre Eq. 10 e Eq. 11 (padrão: média simples).
    mu deve ser o mesmo usado no treino, equilibrando incerteza e diversidade na seleção.
    """
    if not hasattr(model, "get_embedding"):
        raise AttributeError("O modelo precisa implementar get_embedding(x) para usar WAAL.")

    model.eval()
    critic.eval()
    all_scores = []
    eps = 1e-10

    print(f"Calculando score WAAL (incerteza - mu*critico) para {len(unlabeled_indices)} amostras...")
    with torch.no_grad():
        for batch in tqdm(unlabeled_loader):
            images = batch[0].to(device)
            probs = torch.softmax(model(images), dim=-1)  # [Batch, ..., C]

            worst_case = -torch.log(torch.min(probs, dim=-1).values + eps)  # Eq. 10
            uniform_case = -torch.sum(torch.log(probs + eps), dim=-1)  # Eq. 11
            uncertainty = uncertainty_weight * worst_case + (1 - uncertainty_weight) * uniform_case
            while uncertainty.dim() > 1:
                uncertainty = torch.mean(uncertainty, dim=-1)

            features = model.get_embedding(images)
            diversity = critic(features)

            all_scores.extend((uncertainty - mu * diversity).cpu().numpy())

    all_scores = np.array(all_scores)
    # Algorithm 1 (Querying Stage): escolhe as amostras com a MENOR pontuação
    selected_positions = np.argsort(all_scores)[:budget]
    return np.array(unlabeled_indices)[selected_positions]
