# Random Oversampling / Random Undersampling

**Categoria:** Data re-balancing
**Arquivo:** `Balance_Strategies/balance_strategies.py`
**Referência:** técnicas clássicas de reamostragem, amplamente descritas em He & Garcia, 2009, *"Learning from Imbalanced Data"* (IEEE TKDE), e citadas na survey *A Comprehensive Survey on Imbalanced Data Learning* (Gao et al., 2025).

## Ideia central

Duas estratégias complementares que igualam a contagem de amostras por classe **sem olhar features nem similaridade** — apenas rótulo e um gerador aleatório:

- **Random Oversampling**: duplica aleatoriamente candidatos de classes minoritárias até que cada classe tenha (aproximadamente) o mesmo número de amostras.
- **Random Undersampling**: remove aleatoriamente candidatos de classes majoritárias até reduzir cada classe ao mesmo número.

São as formas mais simples de data re-balancing: rápidas, sem hiperparâmetros de vizinhança, mas também as mais "cegas" — não usam nenhuma informação sobre onde cada amostra está no espaço de features.

## Algoritmo

**Random Oversampling** (`random_oversample_strategy`):
1. Agrupa candidatos por classe.
2. `target_count` = contagem informada, ou a contagem da classe **majoritária** por padrão.
3. Para cada classe com menos que `target_count` candidatos, sorteia (com reposição) candidatos adicionais dentro do próprio grupo até atingir `target_count`.

**Random Undersampling** (`random_undersample_strategy`):
1. Agrupa candidatos por classe.
2. `target_count` = contagem informada, ou a contagem da classe **minoritária** por padrão.
3. Para cada classe com mais que `target_count` candidatos, sorteia (sem reposição) exatamente `target_count` candidatos do grupo.

## Assinatura

```python
random_oversample_strategy(candidates, target_count=None, random_state=42) -> list
random_undersample_strategy(candidates, target_count=None, random_state=42) -> list
```

- `candidates`: lista de 3-tuplas `(score, index, label)` — mesmo formato de `class_balance_strategy`.
- `target_count`: tamanho-alvo por classe (opcional; ver padrões acima).
- `random_state`: semente do gerador (`numpy.random.default_rng`), para reprodutibilidade.

**Atenção:** diferente das demais estratégias de balanceamento, `random_oversample_strategy` pode devolver o **mesmo candidato repetido** (mesmo índice de dataset mais de uma vez na lista de saída) — isso é esperado: o `Subset`/`DataLoader` vai ler essa amostra mais de uma vez durante o treino.

## Quando usar

- **Oversampling**: quando o número absoluto de amostras disponíveis é pequeno e não se pode dar ao luxo de descartar dados; o custo é redundância (mesma imagem repetida no treino), que pode levar a overfitting nas poucas amostras minoritárias.
- **Undersampling**: quando há dados de sobra na classe majoritária e o custo computacional do treino importa; o custo é perder informação potencialmente útil das amostras descartadas.

## Limitações

Nenhuma das duas usa a estrutura do espaço de features — ao contrário de Tomek Links, NearMiss, SMOTE e ADASYN, que decidem *quais* amostras duplicar/remover/gerar com base em vizinhança. Isso as torna mais rápidas e simples, mas também menos "inteligentes": oversampling pode reforçar ruído (duplicar um outlier), e undersampling pode remover por acaso justamente as amostras mais informativas da classe majoritária (as que ficam perto da fronteira de decisão).
