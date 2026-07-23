def _class_key(label):
    """Chave hasheável pra agrupar por classe: funciona com valor escalar ou tensor multi-posição
    (ex.: sequência de dígitos do SVHN, usada como tupla)."""
    value = label.tolist() if hasattr(label, "tolist") else label
    return tuple(value) if isinstance(value, list) else value


def class_balance_strategy(candidates, max_per_class=None):
    """
    Balanceia uma lista de candidatos por classe, mantendo os de maior score primeiro em cada classe.

    candidates: lista de (score, index, label) — ex.: (confiança, índice no dataset, pseudo-rótulo
    previsto), como em Labeling_Strategies.labeling_strategies.pseudo_labeling_strategy. O score só desempata
    a ordem dentro da classe, não decide quem entra.

    Se max_per_class não for informado, cada classe é limitada à contagem da classe com MENOS
    candidatos (balanceamento estrito e automático) — evita que uma classe super-representada (a que
    o modelo mais acerta, logo mais confiante) domine a seleção. Se informado, cada classe é limitada
    a min(contagem_da_classe, max_per_class).

    Devolve a lista de candidatos balanceada, no mesmo formato de entrada (score, index, label).
    """
    if not candidates:
        return []

    candidates_by_class = {}
    for score, index, label in candidates:
        candidates_by_class.setdefault(_class_key(label), []).append((score, index, label))

    quota = max_per_class if max_per_class is not None else min(len(group) for group in candidates_by_class.values())

    balanced = []
    for group in candidates_by_class.values():
        group.sort(key=lambda c: c[0], reverse=True)  # mais confiantes primeiro
        balanced.extend(group[:quota])

    return balanced
