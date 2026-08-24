# Tomek Links (`tomek_links_strategy`)

**Categoria:** Data re-balancing (undersampling por vizinhança)
**Arquivo:** `Balance_Strategies/balance_strategies.py`
**Referência:** Tomek, I. (1976). *"Two Modifications of CNN"*. IEEE Transactions on Systems, Man, and Cybernetics — extensão do Condensed Nearest Neighbor de Hart (1968).

## Ideia central

Um **Tomek Link** é um par de amostras `(a, b)` de classes diferentes onde `a` é o vizinho mais próximo de `b` e, ao mesmo tempo, `b` é o vizinho mais próximo de `a` (vizinhança mútua). Esses pares marcam pontos exatamente na fronteira de decisão entre as duas classes — ou ruído (rótulo errado), ou uma região de sobreposição real entre classes.

A estratégia remove o membro **majoritário** de cada Tomek Link encontrado, mantendo o minoritário intacto. O efeito é "limpar" a fronteira: elimina amostras majoritárias que invadem o território da classe minoritária, tornando a fronteira de decisão mais nítida — sem gerar nenhuma amostra nova e sem alterar a classe minoritária.

## Algoritmo

1. Separa os candidatos em classe **majoritária** (maior contagem) e **minoritária** (todas as demais classes juntas — ver `_split_majority_minority`).
2. Para cada amostra majoritária, encontra seu vizinho mais próximo (1-NN) dentro da minoritária.
3. Para cada amostra minoritária, encontra seu vizinho mais próximo (1-NN) dentro da majoritária.
4. Um par `(maj, min)` é um Tomek Link se o vizinho mais próximo de `maj` é `min` **e** o vizinho mais próximo de `min` é `maj` (mutualidade).
5. Remove todas as amostras majoritárias que participam de algum Tomek Link; mantém a minoritária inteira.

## Assinatura

```python
tomek_links_strategy(candidates, **kwargs) -> list
```

- `candidates`: lista de **4-tuplas** `(score, index, label, features)` — diferente de `class_balance_strategy`/`random_oversample_strategy`, aqui é obrigatório fornecer `features` (por exemplo, `model.get_embedding(x)`), já que a decisão depende de distância no espaço de embedding.
- Retorno: lista combinada (majoritária filtrada + minoritária inteira), no mesmo formato 4-tupla de entrada.

## Quando usar

Como método isolado, Tomek Links tipicamente remove **poucas** amostras (só as que estão exatamente na fronteira), então raramente resolve um desbalanceamento forte sozinho. Na literatura, costuma ser combinado com uma técnica de oversampling (ex.: SMOTE + Tomek Links) — o oversampling aumenta a minoritária, e o Tomek Links limpa a sobreposição resultante entre as classes.

## Limitações

- Não reduz drasticamente a classe majoritária (o número de Tomek Links tende a ser pequeno).
- Custo `O(n log n)` por causa da busca de vizinho mais próximo (via `sklearn.neighbors.NearestNeighbors`), repetida duas vezes (majoritária→minoritária e minoritária→majoritária).
- Assume um cenário binário (uma classe majoritária vs. "o resto"); com múltiplas classes minoritárias distintas, todas são tratadas como um único grupo agregado.
