import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# MPL — Meta Pseudo Labels (Pham, Dai, Xie, Luong & Le, 2021, CVPR, "Meta Pseudo Labels"): diferente
# de Pseudo-Label clássico (onde o teacher é FIXO), aqui o teacher é treinado EM PARALELO com o
# student, usando como sinal de recompensa o quanto os pseudo-rótulos do teacher ajudaram o student a
# melhorar num batch rotulado — uma otimização em dois níveis (bi-level): o student otimiza a loss
# não supervisionada (com pseudo-rótulos do teacher), o teacher otimiza a loss supervisionada DO
# STUDENT, através da atualização do student. Técnica de TREINO (loop próprio); precisa de DOIS
# modelos (teacher e student) com a mesma arquitetura mas pesos independentes.


def train_step_mpl(teacher, student, teacher_optimizer, student_optimizer,
                    labeled_batch, unlabeled_batch, task, device, temperature=1.0):
    """
    Um passo de Meta Pseudo Labels (Eqs. 1-3 do paper, versão prática com rótulos "hard" — Seção 2,
    "we sample the hard pseudo labels... rely on a modified version of REINFORCE"):

    1. Teacher prediz no batch não rotulado e AMOSTRA um pseudo-rótulo hard de sua distribuição
       (categorical sample, não argmax — necessário pro REINFORCE).
    2. Student dá UM passo de gradiente real usando esses pseudo-rótulos:
           θ'_S = θ_S - η_S·∇_θ_S L_u(θ_T,θ_S),   L_u = CE(pseudo_labels, S(x_u;θ_S))
    3. Mede a loss supervisionada do student (JÁ ATUALIZADO) num batch ROTULADO — a diferença em
       relação à loss ANTES do passo é a "recompensa": negativa (loss caiu) = pseudo-rótulo ajudou.
    4. Teacher é atualizado via REINFORCE aproximado, reforçando (ou penalizando) a probabilidade que
       ele deu ao pseudo-rótulo amostrado, proporcionalmente à recompensa:

           reward = L_l(θ_S) - L_l(θ'_S)
           teacher_loss = -reward · log T(ŷ_u | x_u; θ_T)                  (política REINFORCE)

    Requer task com atributo .criterion. Devolve um dict com as losses do passo (pra logging).
    """
    if not hasattr(task, "criterion"):
        raise AttributeError("MPL exige uma task com atributo .criterion (ex.: ClassificationTask).")

    x_l, y_l = labeled_batch[0].to(device), labeled_batch[1].to(device)
    x_u = unlabeled_batch[0].to(device)

    # 1) Teacher prediz e amostra um pseudo-rótulo hard (estocástico, não argmax)
    teacher.eval()
    with torch.no_grad():
        teacher_logits_u = teacher(x_u)
    teacher_probs_u = torch.softmax(teacher_logits_u / temperature, dim=-1)
    flat_probs = teacher_probs_u.reshape(-1, teacher_probs_u.size(-1))
    pseudo_labels = torch.multinomial(flat_probs, num_samples=1).squeeze(-1).reshape(teacher_probs_u.shape[:-1])

    # loss do student ANTES do passo, no batch rotulado (linha de base pra recompensa)
    student.eval()
    with torch.no_grad():
        loss_l_before = task.criterion(student(x_l), y_l).item()

    # 2) Student dá um passo real usando os pseudo-rótulos do teacher
    student.train()
    student_optimizer.zero_grad()
    student_logits_u = student(x_u)
    loss_u = F.cross_entropy(
        student_logits_u.reshape(-1, student_logits_u.size(-1)), pseudo_labels.reshape(-1)
    )
    loss_u.backward()
    student_optimizer.step()

    # 3) loss do student DEPOIS do passo, no mesmo batch rotulado -> recompensa
    student.eval()
    with torch.no_grad():
        loss_l_after = task.criterion(student(x_l), y_l).item()
    reward = loss_l_before - loss_l_after

    # 4) Atualiza o teacher via REINFORCE aproximado
    teacher.train()
    teacher_optimizer.zero_grad()
    teacher_logits_u_grad = teacher(x_u)
    log_prob = -F.cross_entropy(
        teacher_logits_u_grad.reshape(-1, teacher_logits_u_grad.size(-1)), pseudo_labels.reshape(-1),
        reduction="none",
    )
    teacher_loss = -(reward * log_prob).mean()
    teacher_loss.backward()
    teacher_optimizer.step()

    return {
        "loss_u": loss_u.item(),
        "teacher_loss": teacher_loss.item(),
        "reward": reward,
        "loss_l_before": loss_l_before,
        "loss_l_after": loss_l_after,
    }


def train_model_with_mpl(
    teacher, student, labeled_loader, unlabeled_loader, test_loader, task,
    teacher_optimizer, student_optimizer, device, epochs, temperature=1.0,
    writer=None, log_prefix="",
):
    """Loop de treino completo do MPL: chama train_step_mpl a cada par de batches (rotulado, não
    rotulado), reciclando o loader menor. Avalia o STUDENT (é ele quem se quer usar no final — o
    teacher existe só pra gerar pseudo-rótulos melhores)."""
    loss_val = 0.0
    test_metric = 0.0

    for epoch in range(epochs):
        running_loss_u = 0.0
        running_reward = 0.0
        n_steps = 0

        unlabeled_iter = iter(unlabeled_loader)
        progress_bar = tqdm(labeled_loader, desc=f"MPL [{epoch + 1}/{epochs}]", unit="batch", leave=False)
        for labeled_batch in progress_bar:
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)

            stats = train_step_mpl(teacher, student, teacher_optimizer, student_optimizer,
                                    labeled_batch, unlabeled_batch, task, device, temperature)

            running_loss_u += stats["loss_u"]
            running_reward += stats["reward"]
            n_steps += 1
            progress_bar.set_postfix(loss_u=f"{stats['loss_u']:.3f}", reward=f"{stats['reward']:.4f}")

        loss_val = running_loss_u / n_steps
        avg_reward = running_reward / n_steps
        test_metric = _evaluate(student, test_loader, task, device, epoch, epochs)
        print(f"Epoch {epoch + 1}/{epochs}: Loss_u {loss_val:.4f} | Avg Reward {avg_reward:.4f} | "
              f"Test Metric (student) {test_metric:.4f}")

        if writer is not None:
            writer.add_scalar(f"{log_prefix}Loss/train_u", loss_val, epoch)
            writer.add_scalar(f"{log_prefix}Reward/train", avg_reward, epoch)
            writer.add_scalar(f"{log_prefix}Metric/test", test_metric, epoch)

    return loss_val, test_metric


def finetune_student_on_labeled(student, labeled_loader, task, optimizer, device, steps):
    """
    "Since the student in Meta Pseudo Labels only learns from unlabeled data with pseudo labels...
    we can take a student model that has converged after training with Meta Pseudo Labels and
    finetune it on labeled data" — passo final opcional do paper, um fine-tuning curto e direto
    (supervisionado padrão) só nos dados rotulados de verdade, pra corrigir qualquer viés herdado dos
    pseudo-rótulos.
    """
    student.train()
    labeled_iter = iter(labeled_loader)
    progress_bar = tqdm(range(steps), desc="Finetune", unit="step")
    for _ in progress_bar:
        try:
            batch = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            batch = next(labeled_iter)

        x, y = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        loss = task.criterion(student(x), y)
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return student


def _evaluate(model, loader, task, device, epoch, epochs):
    model.eval()
    all_metrics = []
    progress_bar = tqdm(loader, desc=f"Eval [{epoch + 1}/{epochs}]", unit="batch", leave=False)
    for batch in progress_bar:
        all_metrics.extend(task.compute_metric(model, batch, device))
    return float(np.mean(all_metrics))
