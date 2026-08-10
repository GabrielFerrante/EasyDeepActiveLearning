import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# VAT — Virtual Adversarial Training (Miyato, Maeda, Koyama & Ishii, 2018, TPAMI, "Virtual
# Adversarial Training: a Regularization Method for Supervised and Semi-Supervised Learning"):
# técnica de TREINO (loop próprio, substitui Training/train.py). Diferente de FixMatch/UDA (que usam
# augmentation de imagem como "ruído"), VAT calcula a PERTURBAÇÃO ADVERSARIAL — a direção no espaço
# de entrada que mais muda a predição do modelo — e treina o modelo pra ser consistente mesmo sob
# essa perturbação, sem precisar de rótulo (por isso funciona em dados rotulados E não rotulados).


def _l2_normalize(d):
    d_flat = d.view(d.size(0), -1)
    norm = d_flat.norm(dim=1, keepdim=True) + 1e-8
    return (d_flat / norm).view_as(d)


def compute_vat_perturbation(model, x, epsilon=8.0, xi=1e-6, power_iterations=1):
    """
    Aproxima r_vadv via power iteration (Eqs. 9-15 do paper): r_vadv é o autovetor dominante da
    Hessiana da divergência D em torno de r=0, aproximado sem calcular a Hessiana explicitamente —
    usa diferença finita (perturbação de tamanho xi) pra estimar Hd via ∇_r D(·)|_{r=ξd}, repetido
    power_iterations vezes (o paper mostra que 1 iteração já é suficiente na prática):

        d ← normalize(∇_r D[p(y|x,θ̂), p(y|x+r,θ̂)]|_{r=ξd}),  repetido power_iterations vezes
        r_vadv ≈ ε · d

    onde θ̂ é o modelo atual tratado como FIXO durante a busca (sem gradiente pros parâmetros — só
    pra direção d). O modelo é temporariamente colocado em eval() durante a busca, pra BatchNorm não
    atualizar suas estatísticas com as entradas perturbadas (não abordado explicitamente no paper,
    mas prática padrão em implementações modernas com BatchNorm).

    Devolve (r_vadv, ref_probs) — ref_probs = p(y|x,θ̂) já computado, reaproveitável pra montar a LDS
    sem uma chamada extra ao modelo.

    Nota prática sobre xi: como xi é usado só pra achar a DIREÇÃO (o resultado final é normalizado e
    reescalado por epsilon), seu valor exato não deveria importar matematicamente — mas na prática,
    xi extremamente pequeno (1e-6, o padrão do paper) pode fazer o gradiente da busca ficar abaixo da
    precisão útil do float32, especialmente em modelos pouco treinados ou com entradas em escala
    grande (ex.: pixels em [0,255] em vez de normalizados). Se `r_vadv.norm()` sair muito menor que
    epsilon, aumente xi (ex.: 1e-2) até a norma se aproximar de epsilon.
    """
    training_mode = model.training
    model.eval()
    try:
        with torch.no_grad():
            ref_probs = torch.softmax(model(x), dim=-1)

        d = _l2_normalize(torch.randn_like(x))
        for _ in range(power_iterations):
            # r = xi*d como a própria folha (leaf) que recebe gradiente — diferenciar em relação a d
            # em vez de r introduziria um fator xi extra pela regra da cadeia (d(x+xi*d)/d(d) = xi),
            # o que faz o gradiente perder precisão numérica pra valores pequenos de xi.
            r = (xi * d).detach().requires_grad_()
            perturbed_probs = torch.softmax(model(x + r), dim=-1)
            dist = F.kl_div(torch.log(perturbed_probs + 1e-8), ref_probs, reduction="batchmean")
            grad = torch.autograd.grad(dist, r)[0]
            d = _l2_normalize(grad).detach()
    finally:
        model.train(training_mode)

    return epsilon * d, ref_probs


def vat_loss(model, x, epsilon=8.0, xi=1e-6, power_iterations=1):
    """
    LDS(x,θ) = D[p(y|x,θ̂), p(y|x+r_vadv,θ)]  (Eq. 5) — a "Local Distributional Smoothness": quão
    diferente a predição do modelo fica ao mover a entrada na direção adversarial. Minimizar essa
    divergência (com gradiente fluindo só pelo termo p(y|x+r_vadv,θ), não pela referência θ̂) é o que
    torna o modelo suave/robusto ao redor de cada ponto — funciona em qualquer x, com ou sem rótulo.
    """
    r_vadv, ref_probs = compute_vat_perturbation(model, x, epsilon, xi, power_iterations)
    perturbed_log_probs = torch.log_softmax(model(x + r_vadv), dim=-1)
    return F.kl_div(perturbed_log_probs, ref_probs, reduction="batchmean")


def train_model_with_vat(
    model, labeled_loader, unlabeled_loader, test_loader, task,
    optimizer, device, epochs, epsilon=8.0, xi=1e-6, power_iterations=1, alpha=1.0,
    writer=None, log_prefix="",
):
    """
    Loop de treino do VAT (Eq. 8): ℓ(D_l,θ) + α·R_vadv(D_l,D_ul,θ), onde ℓ é a NLL supervisionada e
    R_vadv é a média de LDS(x,θ) sobre rotulado E não rotulado (VAT não precisa de rótulo pra
    calcular a perturbação nem a LDS). alpha e epsilon são os únicos dois hiperparâmetros do método —
    o paper reporta que, na prática, só epsilon precisa de ajuste fino (alpha=1 funciona bem).

    Requer task com atributo .criterion.
    """
    if not hasattr(task, "criterion"):
        raise AttributeError("VAT exige uma task com atributo .criterion (ex.: ClassificationTask).")

    loss_val = 0.0
    test_metric = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        n_steps = 0

        unlabeled_iter = iter(unlabeled_loader)
        progress_bar = tqdm(labeled_loader, desc=f"VAT [{epoch + 1}/{epochs}]", unit="batch", leave=False)
        for labeled_batch in progress_bar:
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)

            x_l, y_l = labeled_batch[0].to(device), labeled_batch[1].to(device)
            x_u = unlabeled_batch[0].to(device)

            optimizer.zero_grad()

            outputs_l = model(x_l)
            supervised_loss = task.criterion(outputs_l, y_l)

            x_all = torch.cat([x_l, x_u], dim=0)
            lds = vat_loss(model, x_all, epsilon, xi, power_iterations)

            total_loss = supervised_loss + alpha * lds
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            n_steps += 1
            progress_bar.set_postfix(sup=f"{supervised_loss.item():.3f}", lds=f"{lds.item():.3f}")

        loss_val = running_loss / n_steps
        test_metric = _evaluate(model, test_loader, task, device, epoch, epochs)
        print(f"Epoch {epoch + 1}/{epochs}: Loss {loss_val:.4f} | Test Metric: {test_metric:.4f}")

        if writer is not None:
            writer.add_scalar(f"{log_prefix}Loss/train", loss_val, epoch)
            writer.add_scalar(f"{log_prefix}Metric/test", test_metric, epoch)

    return loss_val, test_metric


def _evaluate(model, loader, task, device, epoch, epochs):
    model.eval()
    all_metrics = []
    progress_bar = tqdm(loader, desc=f"Eval [{epoch + 1}/{epochs}]", unit="batch", leave=False)
    for batch in progress_bar:
        all_metrics.extend(task.compute_metric(model, batch, device))
    return float(np.mean(all_metrics))
