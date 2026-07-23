import os
import csv
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from train import train_model


def save_al_checkpoint(model, optimizer, cycle, labeled_indices, metric, path="checkpoints"):
    os.makedirs(path, exist_ok=True)
    checkpoint = {
        'cycle': cycle,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'labeled_indices': labeled_indices,
        'test_metric': metric,
    }
    filename = os.path.join(path, f"model_cycle_{cycle}_metric_{metric:.2f}.pt")
    torch.save(checkpoint, filename)
    print(f"Checkpoint salvo: {filename}")


def _append_history(csv_path, cycle, num_samples, loss, metric):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer_csv = csv.writer(f)
        if not file_exists:
            writer_csv.writerow(['cycle', 'num_samples', 'final_loss', 'test_metric'])
        writer_csv.writerow([cycle, num_samples, loss, metric])


def run_active_learning_cycle(
    model_fn,
    dataset,
    test_loader,
    task,
    query_strategy,
    labeled_indices,
    unlabeled_indices,
    optimizer_fn,
    device,
    num_cycles,
    cycle_budget,
    epochs,
    batch_size=32,
    num_workers=4,
    checkpoint_dir="al_checkpoints",
    log_dir="runs",
    history_csv=None,
    pseudo_labeling_fn=None,
    pseudo_dataset_cls=None,
    pseudo_balance_fn=None,
    pseudo_label_confidence=0.95,
    pseudo_label_max_per_class=None,
):
    """
    Orquestra N ciclos de active learning: a cada ciclo (re)treina o modelo do zero (via model_fn) só
    com o conjunto rotulado (+ pseudo-rotulado, se pseudo_labeling_fn for informado), avalia em
    test_loader, roda a query_strategy escolhida sobre o pool não rotulado e move as amostras
    selecionadas de unlabeled_indices para labeled_indices.

    model_fn: callable sem argumentos que devolve um modelo novo (já inicializado) a cada ciclo.
    task: instância de uma classe de Training/tasks.py (Classification/Segmentation/Detection/
    VLMContrastive) — define loss, métrica e incerteza para a tarefa em questão.
    optimizer_fn: callable que recebe o modelo do ciclo e devolve o optimizer.
    query_strategy: uma das funções de Query_Strategies/querys_strategies.py (ou compatível com a
    mesma assinatura de kwargs: model, device, budget, unlabeled_loader, unlabeled_indices,
    labeled_loader, labeled_indices, task).

    pseudo_labeling_fn: opcional, ex. Labeling_Strategies.labeling_strategies.pseudo_labeling_strategy.
    Se informado, ao fim de cada ciclo o modelo recém-treinado é usado para gerar pseudo-rótulos (já
    balanceados por classe, via pseudo_label_confidence/pseudo_label_max_per_class) sobre o pool não
    rotulado restante; esses pseudo-rótulos aumentam o treino do ciclo seguinte, mas nunca são
    promovidos a labeled_indices nem tocam o rótulo real de ninguém — pseudo-rotulação é só para
    amostras que nunca foram rotuladas por um oráculo, e é recalculada do zero a cada ciclo (a
    partir do modelo mais recente), pra não propagar um pseudo-rótulo errado pro resto do treino.
    pseudo_dataset_cls: obrigatório se pseudo_labeling_fn for informado — classe compatível com
    Labeling_Strategies.labeling_strategies.PseudoLabeledDataset(base_dataset, indices, pseudo_labels),
    usada para combinar (via ConcatDataset) o conjunto rotulado real com o pseudo-rotulado do ciclo.
    pseudo_balance_fn: obrigatório se pseudo_labeling_fn for informado — repassado como balance_fn
    pra ele, ex. Balance_Strategies.balance_strategies.class_balance_strategy.
    """
    if pseudo_labeling_fn is not None and (pseudo_dataset_cls is None or pseudo_balance_fn is None):
        raise ValueError("pseudo_labeling_fn foi informado sem pseudo_dataset_cls e/ou pseudo_balance_fn.")

    labeled_indices = np.array(labeled_indices)
    unlabeled_indices = np.array(unlabeled_indices)
    pseudo_indices, pseudo_labels = np.array([], dtype=int), []

    for cycle in range(num_cycles):
        print(f"\n--- Iniciando Ciclo de AL {cycle} | Labeled Size: {len(labeled_indices)} "
              f"| Pseudo-Labeled Size: {len(pseudo_indices)} ---")

        model = model_fn().to(device)
        optimizer = optimizer_fn(model)

        train_dataset = Subset(dataset, labeled_indices)
        if len(pseudo_indices) > 0:
            train_dataset = ConcatDataset([
                train_dataset,
                pseudo_dataset_cls(dataset, pseudo_indices, pseudo_labels),
            ])

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True,
        )

        writer = SummaryWriter(log_dir=os.path.join(log_dir, f"AL_Cycle_{cycle}"))
        loss, test_metric = train_model(
            model, train_loader, test_loader, task, optimizer, device, epochs, writer=writer,
        )
        writer.close()

        save_al_checkpoint(model, optimizer, cycle, labeled_indices, test_metric, path=checkpoint_dir)
        if history_csv:
            _append_history(history_csv, cycle, len(labeled_indices), loss, test_metric)

        print("--- Selecionando novas amostras ---")
        # shuffle=False: a ordem dos batches precisa bater com a ordem de unlabeled_indices/labeled_indices
        unlabeled_loader = DataLoader(
            Subset(dataset, unlabeled_indices), batch_size=batch_size,
            shuffle=False, num_workers=num_workers,
        )
        labeled_loader = DataLoader(
            Subset(dataset, labeled_indices), batch_size=batch_size,
            shuffle=False, num_workers=num_workers,
        )

        new_indices = query_strategy(
            model=model,
            device=device,
            budget=cycle_budget,
            unlabeled_loader=unlabeled_loader,
            unlabeled_indices=unlabeled_indices,
            labeled_loader=labeled_loader,
            labeled_indices=labeled_indices,
            task=task,
        )

        new_indices_set = set(new_indices)
        unlabeled_indices = np.array([i for i in unlabeled_indices if i not in new_indices_set])
        labeled_indices = np.concatenate([labeled_indices, new_indices])

        pseudo_indices, pseudo_labels = np.array([], dtype=int), []
        if pseudo_labeling_fn is not None and len(unlabeled_indices) > 0:
            print("--- Gerando pseudo-rótulos (balanceados por classe) ---")
            pseudo_unlabeled_loader = DataLoader(
                Subset(dataset, unlabeled_indices), batch_size=batch_size,
                shuffle=False, num_workers=num_workers,
            )
            pseudo_indices, pseudo_labels = pseudo_labeling_fn(
                model=model,
                device=device,
                unlabeled_loader=pseudo_unlabeled_loader,
                unlabeled_indices=unlabeled_indices,
                balance_fn=pseudo_balance_fn,
                confidence_threshold=pseudo_label_confidence,
                max_per_class=pseudo_label_max_per_class,
            )
            print(f"{len(pseudo_indices)} amostras pseudo-rotuladas para o próximo ciclo.")

    return labeled_indices, unlabeled_indices
