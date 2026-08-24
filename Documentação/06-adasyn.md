# ADASYN — Adaptive Synthetic Sampling (`adasyn_oversample`)

**Categoria:** Data re-balancing (geração sintética adaptativa)
**Arquivo:** `Balance_Strategies/balance_strategies.py`
**Referência:** He, H., Bai, Y., Garcia, E. A. & Li, S. (2008). *"ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning"*. IEEE International Joint Conference on Neural Networks (IJCNN).

## Ideia central

ADASYN é uma evolução do SMOTE: em vez de gerar o mesmo número de sintéticas para cada amostra minoritária, gera **mais sintéticas para as amostras "difíceis"** — aquelas perto da fronteira de decisão, cercadas por vizinhos de outras classes. A intuição é que essas regiões são onde o classificador mais erra, então merecem mais reforço sintético; amostras minoritárias já "seguras" (cercadas só por vizinhos da própria classe) recebem poucas ou nenhuma sintética.

## Algoritmo

Para cada amostra minoritária `i`:

1. Calcula `r_i = (número de vizinhos de OUTRA classe) / k`, olhando os `k` vizinhos mais próximos entre **todos** os candidatos (não só da própria classe) — essa é a medida de "quão perto da fronteira" a amostra está.
2. Normaliza `r_i` para somar 1 dentro da classe: `r̂_i = r_i / Σr_i`.
3. Distribui o total de sintéticas necessárias (`n_needed = target_count - len(classe)`) proporcionalmente a `r̂_i`: `g_i = round(r̂_i · n_needed)`.
4. Para cada amostra `i`, gera `g_i` sintéticas por interpolação — **igual ao SMOTE**: escolhe um vizinho `j` da mesma classe, sorteia `λ ~ U(0,1)`, e interpola `x_new = x_i + λ·(x_j - x_i)`.

A única diferença real em relação ao SMOTE é *quantas* sintéticas cada amostra-base recebe (proporcional à dificuldade, via `r_i`), não *como* a interpolação em si é feita.

## Assinatura

```python
adasyn_oversample(candidates, target_count=None, k_neighbors=5, random_state=42) -> (candidates, synthetic)
```

- `candidates`: lista de 4-tuplas `(score, index, label, features)`.
- `k_neighbors`: usado tanto para calcular `r_i` (vizinhança global, entre todas as classes) quanto para a interpolação (vizinhança local, dentro da própria classe) — os dois usos são independentes internamente (`k_global` e `k_local`).
- Retorno: mesma interface de `smote_oversample` — `(candidates, synthetic)`, com `synthetic` = lista de `(label, embedding_sintético)`.

Usa a mesma infraestrutura de treino que SMOTE: `SyntheticEmbeddingDataset` + `EmbeddingClassifierWrapper(model)` (exige `model.classify_from_embedding(features)`).

## Quando usar

Quando, além de balancear a contagem de classes, também importa que o classificador receba reforço extra exatamente nas regiões onde ele mais confunde as classes — ADASYN concentra o esforço sintético ali, enquanto SMOTE distribui uniformemente entre todas as amostras minoritárias.

## Limitações

- Herdando do SMOTE: qualidade depende do espaço de embedding usado.
- Se `r_i` for zero para todas as amostras de uma classe (nenhuma tem vizinho de outra classe entre os `k_global` mais próximos — classe já bem separada), a distribuição cai para uniforme (`r = 1/len(grupo)` para todas), similar ao SMOTE nesse caso.
- Mais caro computacionalmente que SMOTE: exige uma busca de vizinhos **global** (entre todos os candidatos) além da busca local por classe.
