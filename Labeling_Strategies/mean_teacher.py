import numpy as np
import torch
from tqdm import tqdm

from augmentation import ema_update

# Mean Teacher (Tarvainen & Valpola, 2017, NeurIPS, "Mean teachers are better role models: Weight-
# averaged consistency targets improve semi-supervised deep learning results"): técnica de TREINO
# (loop próprio, substitui Training/train.py). Mantém um segundo modelo (teacher) cujos pesos são a
# média móvel exponencial (EMA) dos pesos do modelo sendo treinado (student) — o teacher nunca recebe
# gradiente diretamente, só a atualização EMA depois de cada passo do student.


def _rampup(weight, step, rampup_steps):
    if not rampup_steps:
        return weight
    return weight * min(1.0, step / rampup_steps)


def train_model_with_mean_teacher(
    student, teacher, labeled_loader, unlabeled_loader, test_loader, task,
    optimizer, device, epochs, ema_decay=0.999, consistency_weight=1.0,
    consistency_rampup_steps=None, writer=None, log_prefix="",
):
    """
    Loop de treino do Mean Teacher. A cada passo:

        θ'_t = ema_decay · θ'_{t-1} + (1 - ema_decay) · θ_t                — EMA do teacher (Eq. sem número)
        J(θ) = E[ ||f(x,θ',η') - f(x,θ,η)||² ]                            — consistency loss (MSE)
        loss = classification_cost(student) + consistency_weight · J(θ)

    O "ruído" η/η' do paper (augmentation/dropout independentes pro student e o teacher) vem aqui do
    próprio dropout de cada modelo em model.train() — mesma imagem, duas passadas (student com
    gradiente, teacher sem) — uma simplificação comum de implementações públicas do método, que
    dispensa augmentation dupla explícita.

    A consistency loss é aplicada tanto no rotulado quanto no não rotulado (como no Fig. 2 do paper);
    a classification loss só no rotulado. teacher: uma cópia do student (ver
    Labeling_Strategies.augmentation.clone_model) — nunca é otimizado diretamente, só via EMA.

    Requer task com atributo .criterion.

    consistency_rampup_steps: se informado, o peso da consistency loss sobe linearmente de 0 até
    consistency_weight ao longo desses passos — o paper usa um ramp-up (sigmoide) pelos primeiros
    passos, já que no início o teacher (recém-inicializado) não tem alvos confiáveis.
    """
    if not hasattr(task, "criterion"):
        raise AttributeError("Mean Teacher exige uma task com atributo .criterion (ex.: ClassificationTask).")

    total_step = 0
    loss_val = 0.0
    test_metric = 0.0

    for epoch in range(epochs):
        student.train()
        teacher.train()  # mantém dropout ativo no teacher também (fonte do "ruído" η')
        running_loss = 0.0
        n_steps = 0

        unlabeled_iter = iter(unlabeled_loader)
        progress_bar = tqdm(labeled_loader, desc=f"MeanTeacher [{epoch + 1}/{epochs}]", unit="batch", leave=False)
        for labeled_batch in progress_bar:
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)

            x_l, y_l = labeled_batch[0].to(device), labeled_batch[1].to(device)
            x_u = unlabeled_batch[0].to(device)

            optimizer.zero_grad()

            outputs_l = student(x_l)
            classification_loss = task.criterion(outputs_l, y_l)

            x_all = torch.cat([x_l, x_u], dim=0)
            student_probs = torch.softmax(student(x_all), dim=-1)
            with torch.no_grad():
                teacher_probs = torch.softmax(teacher(x_all), dim=-1)
            consistency_loss = ((student_probs - teacher_probs) ** 2).mean()

            weight = _rampup(consistency_weight, total_step, consistency_rampup_steps)
            total_loss = classification_loss + weight * consistency_loss
            total_loss.backward()
            optimizer.step()

            ema_update(teacher, student, ema_decay)

            running_loss += total_loss.item()
            n_steps += 1
            total_step += 1
            progress_bar.set_postfix(cls=f"{classification_loss.item():.3f}",
                                      cons=f"{consistency_loss.item():.3f}", w=f"{weight:.2f}")

        loss_val = running_loss / n_steps
        test_metric = _evaluate(teacher, test_loader, task, device, epoch, epochs)
        print(f"Epoch {epoch + 1}/{epochs}: Loss {loss_val:.4f} | Test Metric (teacher) {test_metric:.4f}")

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
