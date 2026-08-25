import os
import csv
import time
from contextlib import contextmanager

import numpy as np
import torch

# Diferente de Metrics/task_metrics.py (que avalia a TAREFA em si — accuracy, IoU, mAP, ...), este
# módulo mede o PROCESSO de active learning: quanto tempo cada fase de um ciclo consome, quão
# informativas as amostras escolhidas pela query eram (o modelo já acertava, ou eram genuinamente
# difíceis?), e quantas classes distintas já são conhecidas no conjunto rotulado a cada ciclo.


class Timer:
    """Cronômetro via context manager: `with Timer() as t: ...` deixa o tempo decorrido (segundos) em
    `t.elapsed` depois do bloco."""

    def __enter__(self):
        self._start = time.perf_counter()
        self.elapsed = None
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self._start
        return False


def _flatten_labels(labels):
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    return np.asarray(labels).reshape(-1)


def count_known_classes(labels):
    """
    Quantas classes distintas aparecem em `labels` — os rótulos de todo o conjunto rotulado atual.
    Achata qualquer formato (rótulo escalar por amostra, ou multi-posição como a sequência de dígitos
    do SVHN) e conta valores distintos; para saídas multi-posição, conta a união de classes vistas em
    QUALQUER posição, não uma contagem separada por posição.
    """
    return int(np.unique(_flatten_labels(labels)).size)


def update_known_classes(known_classes, labels):
    """
    Atualiza incrementalmente um set() de classes conhecidas com os rótulos de um novo lote de
    amostras (ex.: as recém-rotuladas/aceitas num ciclo de AL) — mais barato que recontar o conjunto
    rotulado inteiro a cada ciclo, já que só as amostras NOVAS podem introduzir classe nova. Devolve o
    mesmo set (mutado in-place), pra encadear: known = update_known_classes(known, new_labels).
    """
    known_classes.update(_flatten_labels(labels).tolist())
    return known_classes


def selected_sample_metric(model, device, task, loader):
    """
    Mede o desempenho do modelo ATUAL (antes de re-treinar neste ciclo) sobre um loader pequeno —
    tipicamente só as amostras que a query strategy acabou de escolher neste ciclo, uma vez que o
    rótulo delas se torna conhecido (via oráculo real ou pseudo-labeling). Quantifica "quão
    informativas" as amostras selecionadas realmente eram: se o modelo já acertava a maioria, a
    seleção não trouxe muita informação nova pro treino seguinte; se errava muito, capturou pontos
    genuinamente difíceis/incertos — o comportamento esperado de uma boa query strategy de incerteza.

    Reaproveita task.compute_metric (a mesma métrica por amostra já configurada na task — accuracy pra
    classificação, IoU pra segmentação, recall pra detecção, MAE-based pra regressão, retrieval top-1
    pra VLM) em vez de reimplementar uma métrica própria, então o número aqui é diretamente comparável
    ao test_metric do ciclo.

    loader: DataLoader (shuffle=False) só com as amostras selecionadas, no formato de batch que a task
    espera (mesma convenção do restante do ciclo). Devolve NaN se o loader estiver vazio (ex.: nenhum
    candidato foi aceito no ciclo).
    """
    model.eval()
    scores = []
    for batch in loader:
        scores.extend(task.compute_metric(model, batch, device))
    return float(np.mean(scores)) if scores else float("nan")


class ALMetricsTracker:
    """
    Acumula, por ciclo, as métricas do PROCESSO de active learning: tempos de seleção/treino/
    classificação, quantidade de classes conhecidas no conjunto rotulado, e o desempenho do modelo
    sobre as amostras que ele mesmo acabou de selecionar (ver selected_sample_metric). Já é criado e
    populado automaticamente por Training/active_learning_cycle.py e Training/self_labeling_cycle.py
    — passe sua própria instância só se quiser inspecionar/plotar os dados depois (o retorno das duas
    funções de ciclo não muda; o tracker vive na instância que você passou).

    Uso manual (fora dos ciclos prontos), se necessário:

        tracker = ALMetricsTracker()
        for cycle in range(num_cycles):
            with tracker.timed(cycle, "training_time_s"):
                ...treina...
            with tracker.timed(cycle, "selection_time_s"):
                new_indices = query_strategy(...)
            tracker.record(cycle, known_classes=..., selected_sample_metric=..., test_metric=...)
        tracker.to_csv("al_metrics.csv")
    """

    def __init__(self):
        self._records = {}

    def _cycle(self, cycle):
        return self._records.setdefault(cycle, {"cycle": cycle})

    @contextmanager
    def timed(self, cycle, key):
        with Timer() as t:
            yield
        self._cycle(cycle)[key] = t.elapsed

    def record(self, cycle, **kwargs):
        self._cycle(cycle).update(kwargs)

    def get(self, cycle):
        return dict(self._records.get(cycle, {}))

    def as_list(self):
        return [self._records[c] for c in sorted(self._records)]

    def to_csv(self, path):
        """Grava o estado ATUAL do tracker inteiro em `path` (sobrescreve — seguro de chamar a cada
        ciclo pra manter um arquivo sempre atualizado, sem risco de duplicar linhas)."""
        rows = self.as_list()
        if not rows:
            return
        fieldnames = sorted({k for row in rows for k in row}, key=lambda k: (k != "cycle", k))
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, mode="w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=fieldnames)
            writer_csv.writeheader()
            for row in rows:
                writer_csv.writerow(row)

    def log_to_tensorboard(self, writer, cycle):
        """Escreve os campos numéricos do ciclo (tempos, known_classes, selected_sample_metric, ...)
        no SummaryWriter já usado pelo ciclo de AL, sob a tag 'ALMetrics/<campo>'."""
        for key, value in self._cycle(cycle).items():
            if key == "cycle" or not isinstance(value, (int, float)):
                continue
            writer.add_scalar(f"ALMetrics/{key}", value, cycle)
