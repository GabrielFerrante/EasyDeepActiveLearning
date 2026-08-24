# Balanceamento por Classe/Quota (`class_balance_strategy`)

**Categoria:** Re-labeling (self-training balanceado)
**Arquivo:** `Balance_Strategies/balance_strategies.py`
**Referência:** técnica de propósito geral (não vem de um paper específico), citada na survey *A Comprehensive Survey on Imbalanced Data Learning* (Gao et al., 2025) como "balanceamento por classe/quota" no contexto de re-labeling.

## Ideia central

É a estratégia de balanceamento mais simples da biblioteca: dado um conjunto de candidatos (por exemplo, saídas de pseudo-labeling), **corta cada classe pelo mesmo tamanho**, mantendo os candidatos de maior score (confiança) primeiro. O objetivo é evitar que uma classe super-representada — tipicamente a que o modelo já acerta mais e, por isso, produz mais candidatos confiantes — domine a seleção que entra no treino.

## Algoritmo

1. Agrupa os candidatos por classe (rótulo).
2. Define uma quota:
   - Se `max_per_class` não for informado, usa a contagem da classe com **menos** candidatos (balanceamento estrito e automático).
   - Se informado, usa `min(contagem_da_classe, max_per_class)` para cada classe.
3. Dentro de cada classe, ordena os candidatos por score (maior primeiro) e mantém apenas os `quota` primeiros.

Não há geração nem descarte aleatório: a única decisão é *quantos* de cada classe entram, e *quais* (os mais confiantes).

## Assinatura

```python
class_balance_strategy(candidates, max_per_class=None) -> list
```

- `candidates`: lista de 3-tuplas `(score, index, label)` — por exemplo, `(confiança, índice no dataset, pseudo-rótulo previsto)`, como as que `Labeling_Strategies.labeling_strategies.pseudo_labeling_strategy` produz.
- `max_per_class`: teto opcional por classe.
- Retorno: subconjunto balanceado, no mesmo formato de entrada `(score, index, label)`.

O `label` pode ser um escalar ou um tensor multi-posição (ex.: sequência de dígitos do SVHN) — a função usa uma chave hasheável (`_class_key`) que trata ambos os casos, convertendo tensores em tuplas.

## Quando usar

É a opção padrão quando não há necessidade de gerar amostras sintéticas nem de olhar a vizinhança no espaço de features — só rótulo e score já bastam. É a estratégia mais barata computacionalmente entre todas as de balanceamento da biblioteca, e a única desenhada especificamente para o fluxo de pseudo-labeling (onde "descartar o excesso" é aceitável, já que sempre há mais candidatos surgindo no próximo ciclo).

## Limitações

- Corta candidatos em vez de gerar novos — se uma classe tem poucos candidatos confiantes, o restante das classes é limitado a esse mesmo número, o que pode desperdiçar muitos candidatos bons de outras classes.
- Não considera a distribuição de features/similaridade — dois candidatos da mesma classe podem ser quase idênticos e ambos serem mantidos, enquanto um candidato mais informativo (mas com score levemente menor) é descartado.
