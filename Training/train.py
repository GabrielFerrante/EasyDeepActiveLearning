import numpy as np
from tqdm import tqdm


def train_one_epoch(model, loader, task, optimizer, device, desc="Train"):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(loader, desc=desc, unit="batch", leave=False)
    for batch in progress_bar:
        optimizer.zero_grad()
        loss = task.compute_loss(model, batch, device)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / len(loader)


def evaluate(model, loader, task, device, desc="Eval"):
    model.eval()
    all_metrics = []
    progress_bar = tqdm(loader, desc=desc, unit="batch", leave=False)
    for batch in progress_bar:
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
        loss = train_one_epoch(model, train_loader, task, optimizer, device, desc=f"Train [{epoch + 1}/{epochs}]")
        test_metric = evaluate(model, test_loader, task, device, desc=f"Eval [{epoch + 1}/{epochs}]")
        print(f"Epoch {epoch + 1}/{epochs}: Loss {loss:.4f} | Test Metric: {test_metric:.4f}")

        if writer is not None:
            writer.add_scalar(f"{log_prefix}Loss/train", loss, epoch)
            writer.add_scalar(f"{log_prefix}Metric/test", test_metric, epoch)

    return loss, test_metric
