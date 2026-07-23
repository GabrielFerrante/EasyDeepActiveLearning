import numpy as np


def train_one_epoch(model, loader, task, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        loss = task.compute_loss(model, batch, device)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)


def evaluate(model, loader, task, device):
    model.eval()
    all_metrics = []
    for batch in loader:
        all_metrics.extend(task.compute_metric(model, batch, device))
    return float(np.mean(all_metrics))


def train_model(model, train_loader, test_loader, task, optimizer, device, epochs, writer=None, log_prefix=""):
    """
    Loop de treino genérico: a task (Training/tasks.py) define como calcular loss e métrica para a
    tarefa em questão (Classification, Segmentation, Detection, VLM contrastivo, ...) — o loop em si
    não sabe nada sobre o formato de saída do modelo, apenas delega para a task.
    """
    loss = 0.0
    test_metric = 0.0
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, task, optimizer, device)
        test_metric = evaluate(model, test_loader, task, device)
        print(f"Epoch {epoch + 1}/{epochs}: Loss {loss:.4f} | Test Metric: {test_metric:.4f}")

        if writer is not None:
            writer.add_scalar(f"{log_prefix}Loss/train", loss, epoch)
            writer.add_scalar(f"{log_prefix}Metric/test", test_metric, epoch)

    return loss, test_metric
