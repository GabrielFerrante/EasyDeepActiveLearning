import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from train import train_model, evaluate
from tasks import ClassificationTask
# Metrics/ precisa estar no sys.path do chamador, igual às demais pastas da lib (ex.: Training/).
from al_metrics import ALMetricsTracker, update_known_classes


def _save_checkpoint(model, optimizer, cycle, num_labeled, metric, path):
    os.makedirs(path, exist_ok=True)
    checkpoint = {
        'cycle': cycle,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'num_labeled': num_labeled,
        'test_metric': metric,
    }
    filename = os.path.join(path, f"model_cycle_{cycle}_metric_{metric:.2f}.pt")
    torch.save(checkpoint, filename)
    print(f"Checkpoint salvo: {filename}")


def _append_history(csv_path, cycle, num_labeled, num_accepted, loss, metric):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer_csv = csv.writer(f)
        if not file_exists:
            writer_csv.writerow(['cycle', 'num_labeled', 'num_accepted_this_cycle', 'final_loss', 'test_metric'])
        writer_csv.writerow([cycle, num_labeled, num_accepted, loss, metric])


def run_self_labeling_cycle(
    model_fn,
    unlabeled_dataset,
    build_labeled_dataset_fn,
    test_loader,
    task,
    query_strategy,
    pseudo_labeling_fn,
    balance_fn,
    unlabeled_indices,
    on_accepted,
    optimizer_fn,
    device,
    num_cycles,
    cycle_budget,
    epochs,
    confidence_threshold=0.5,
    max_per_class=None,
    batch_size=32,
    num_workers=4,
    checkpoint_dir="al_checkpoints",
    log_dir="runs",
    history_csv=None,
    metrics_tracker=None,
    al_metrics_csv=None,
):
    """
    Variante de active_learning_cycle.run_active_learning_cycle para quando não existe oráculo real:
    a query_strategy escolhe candidatos do pool não rotulado (unlabeled_dataset) e o próprio
    pseudo_labeling_fn atua como "oráculo" só para ESSES candidatos — não para o pool inteiro, como em
    run_active_learning_cycle — usando um confidence_threshold mais permissivo (padrão 0.5), já que
    não há nenhuma outra fonte de rótulo disponível para eles.

    Candidatos aceitos (confiança >= confidence_threshold, já balanceados por classe via balance_fn)
    são entregues a on_accepted para persistência (ex.: gravar linhas novas num CSV separado, sem
    tocar os rótulos verdadeiros originais) e removidos de unlabeled_indices; os rejeitados voltam
    para o pool não rotulado e podem ser escolhidos de novo em ciclos futuros.

    O dataset de treino é reconstruído do zero a cada ciclo via build_labeled_dataset_fn (ex.: relendo
    o CSV persistido) — quem decide "o que é rotulado hoje" é sempre a base persistida, nunca um
    estado em memória que se perde entre execuções.

    on_accepted(cycle, accepted_indices, accepted_labels): callback responsável por persistir os
    rótulos aceitos e por qualquer instrumentação desejada (ex.: salvar as imagens escolhidas).

    metrics_tracker: instância opcional de Metrics.al_metrics.ALMetricsTracker (criada internamente
    se não informada — sempre instrumentado). Cada ciclo registra tempo de treino/seleção/
    classificação/pseudo-labeling, known_classes (só pra ClassificationTask), e acceptance_rate —
    fração dos candidatos escolhidos pela query que o pseudo-labeling aceitou como oráculo (não há
    rótulo real aqui pra medir "acerto" no sentido tradicional; acceptance_rate é o análogo possível
    neste cenário — quantos dos "achados" da query o próprio modelo confiou o suficiente pra aceitar).
    al_metrics_csv: caminho opcional pra persistir o tracker inteiro em CSV a cada ciclo (sobrescreve).
    """
    unlabeled_indices = np.array(unlabeled_indices)

    if metrics_tracker is None:
        metrics_tracker = ALMetricsTracker()

    is_classification = isinstance(task, ClassificationTask)
    known_classes = set()

    for cycle in range(num_cycles):
        train_dataset = build_labeled_dataset_fn()
        print(f"\n--- Ciclo {cycle} | Base rotulada atual: {len(train_dataset)} amostras | "
              f"Pool não rotulado: {len(unlabeled_indices)} ---")

        model = model_fn().to(device)
        optimizer = optimizer_fn(model)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True,
        )

        if is_classification and cycle == 0:
            for batch in train_loader:
                update_known_classes(known_classes, batch[1])

        writer = SummaryWriter(log_dir=os.path.join(log_dir, f"Cycle_{cycle}"))
        with metrics_tracker.timed(cycle, "training_time_s"):
            loss, test_metric = train_model(
                model, train_loader, test_loader, task, optimizer, device, epochs, writer=writer,
            )
        # passada de inferência dedicada, só pra medir "tempo de classificação" isoladamente do
        # treino (train_model já avalia a cada época, mas misturado ao treino).
        with metrics_tracker.timed(cycle, "classification_time_s"):
            evaluate(model, test_loader, task, device, desc=f"Classify [{cycle}]")

        _save_checkpoint(model, optimizer, cycle, len(train_dataset), test_metric, path=checkpoint_dir)

        print("--- Selecionando candidatos (query) ---")
        unlabeled_loader = DataLoader(
            Subset(unlabeled_dataset, unlabeled_indices), batch_size=batch_size,
            shuffle=False, num_workers=num_workers,
        )
        with metrics_tracker.timed(cycle, "selection_time_s"):
            queried_indices = query_strategy(
                model=model,
                device=device,
                budget=cycle_budget,
                unlabeled_loader=unlabeled_loader,
                unlabeled_indices=unlabeled_indices,
                task=task,
            )

        print("--- Pseudo-labeling como oráculo (só sobre os candidatos da query) ---")
        queried_loader = DataLoader(
            Subset(unlabeled_dataset, queried_indices), batch_size=batch_size,
            shuffle=False, num_workers=num_workers,
        )
        with metrics_tracker.timed(cycle, "pseudo_labeling_time_s"):
            accepted_indices, accepted_labels = pseudo_labeling_fn(
                model=model,
                device=device,
                unlabeled_loader=queried_loader,
                unlabeled_indices=queried_indices,
                balance_fn=balance_fn,
                confidence_threshold=confidence_threshold,
                max_per_class=max_per_class,
            )

        on_accepted(cycle, accepted_indices, accepted_labels)

        if is_classification and len(accepted_labels) > 0:
            update_known_classes(known_classes, accepted_labels)

        acceptance_rate = len(accepted_indices) / len(queried_indices) if len(queried_indices) > 0 else 0.0
        metrics_tracker.record(
            cycle,
            num_labeled=len(train_dataset),
            num_queried=len(queried_indices),
            num_accepted=len(accepted_indices),
            acceptance_rate=acceptance_rate,
            known_classes=(len(known_classes) if is_classification else None),
            train_loss=loss,
            test_metric=test_metric,
        )
        metrics_tracker.log_to_tensorboard(writer, cycle)
        writer.close()
        if al_metrics_csv:
            metrics_tracker.to_csv(al_metrics_csv)

        accepted_set = set(np.asarray(accepted_indices).tolist())
        unlabeled_indices = np.array([i for i in unlabeled_indices if i not in accepted_set])

        if history_csv:
            _append_history(history_csv, cycle, len(train_dataset), len(accepted_indices), loss, test_metric)

        print(f"--- Ciclo {cycle}: {len(accepted_indices)}/{len(queried_indices)} candidatos aceitos "
              f"(threshold {confidence_threshold}) | {len(queried_indices) - len(accepted_indices)} "
              f"voltaram pro pool não rotulado ---")

    return unlabeled_indices
