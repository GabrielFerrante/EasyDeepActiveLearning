# Tomek Links (`tomek_links_strategy`)

**Categoria:** Data re-balancing (undersampling por vizinhança)
**Arquivo:** `Balance_Strategies/balance_strategies.py`
**Referência:** Tomek, I. (1976). *"Two Modifications of CNN"*. IEEE Transactions on Systems, Man, and Cybernetics — extensão do Condensed Nearest Neighbor de Hart (1968).

## Ideia central

Um **Tomek Link** é um par de amostras `(a, b)` de classes diferentes tal que não existe nenhum outro ponto `z` (de qualquer classe) mais perto de `a` do que `b`, nem mais perto de `b` do que `a` — ou seja, `b` é o vizinho mais próximo **global** de `a` (considerando todos os pontos, de qualquer classe) e vice-versa (vizinhança mútua). Esses pares marcam pontos exatamente na fronteira de decisão entre as duas classes — ou ruído (rótulo errado), ou uma região de sobreposição real entre classes.

A estratégia remove o membro **majoritário** de cada Tomek Link encontrado, mantendo o minoritário intacto. O efeito é "limpar" a fronteira: elimina amostras majoritárias que invadem o território da classe minoritária, tornando a fronteira de decisão mais nítida — sem gerar nenhuma amostra nova e sem alterar a classe minoritária.

## Algoritmo

1. Separa os candidatos em classe **majoritária** (maior contagem) e **minoritária** (todas as demais classes juntas — ver `_split_majority_minority`), mas junta as features de AMBAS num único conjunto pra busca de vizinhança (ver nota abaixo).
2. Para cada ponto do conjunto combinado, encontra seu vizinho mais próximo **global** (1-NN sobre todos os pontos, de qualquer classe, excluindo ele mesmo).
3. Um par `(maj, min)` é um Tomek Link se o vizinho mais próximo global de `maj` é `min` **e** o vizinho mais próximo global de `min` é `maj` (mutualidade) — isso já garante, por construção, que nenhum outro ponto de nenhuma classe está mais perto de nenhum dos dois.
4. Remove todas as amostras majoritárias que participam de algum Tomek Link; mantém a minoritária inteira.

**Importante — precisa de um único k-NN global, não um k-NN por classe**: ajustar o `NearestNeighbors` separadamente para cada classe (buscar o vizinho mais próximo de cada majoritária *apenas dentre* as minoritárias, e vice-versa) não implementa a definição formal de Tomek Link — pode marcar como Tomek Link um par onde, na verdade, existe um terceiro ponto (da mesma classe que um dos dois) mais próximo, que deveria desqualificar o par. Por exemplo: se uma amostra majoritária `a` tem um vizinho majoritário `c` bem mais próximo que a minoritária `b`, o par `(a,b)` não é um Tomek Link — mas um k-NN ajustado só sobre a minoritária nunca "vê" `c` e aceitaria o par incorretamente. A implementação correta ajusta um único `NearestNeighbors(n_neighbors=2)` sobre o conjunto combinado (majoritária+minoritária), o mesmo padrão usado pela implementação de referência do `imbalanced-learn`.

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
