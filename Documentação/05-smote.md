# SMOTE — Synthetic Minority Oversampling Technique (`smote_oversample`)

**Categoria:** Data re-balancing (geração sintética)
**Arquivo:** `Balance_Strategies/balance_strategies.py`
**Referência:** Chawla, N. V., Bowyer, K. W., Hall, L. O. & Kegelmeyer, W. P. (2002). *"SMOTE: Synthetic Minority Over-sampling Technique"*. Journal of Artificial Intelligence Research, 16, 321-357.

## Ideia central

Em vez de duplicar amostras existentes (Random Oversampling) ou apenas escolher quais manter (NearMiss), SMOTE **gera amostras novas** de classes minoritárias por interpolação linear no espaço de features, entre uma amostra real e um de seus vizinhos mais próximos da mesma classe:

```
x_new = x_i + λ·(x_j - x_i),   λ ~ U(0,1)
```

onde `x_i` é uma amostra minoritária e `x_j` é um de seus `k` vizinhos mais próximos **da mesma classe**. O ponto sintético cai em algum lugar no segmento de reta entre os dois — uma forma de "preencher" o espaço da classe minoritária com pontos plausíveis, em vez de repetir os mesmos pontos várias vezes.

## Algoritmo

1. Agrupa candidatos por classe.
2. `target_count` = contagem informada, ou a contagem da classe majoritária por padrão; `n_needed = target_count - len(grupo)` por classe.
3. Para cada classe com `n_needed > 0`, calcula os `k_neighbors` vizinhos mais próximos de cada amostra **dentro da própria classe** (via `sklearn.neighbors.NearestNeighbors`).
4. Repete `n_needed` vezes: sorteia uma amostra-base da classe, sorteia um de seus vizinhos, sorteia `λ ~ U(0,1)`, e interpola os embeddings.

## Assinatura

```python
smote_oversample(candidates, target_count=None, k_neighbors=5, random_state=42) -> (candidates, synthetic)
```

- `candidates`: lista de 4-tuplas `(score, index, label, features)`.
- Retorno: **tupla** `(candidates, synthetic)` — `candidates` é devolvida sem alteração; `synthetic` é uma lista de pares `(label, embedding_sintético)`, **sem índice real no dataset**.

## Particularidade importante da implementação nesta biblioteca

Diferente das demais estratégias, SMOTE não seleciona entre candidatos existentes — ele cria pontos **no espaço de embedding**, que não correspondem a nenhuma imagem real. Para treinar com essas amostras sintéticas:

1. Envolva-as em `SyntheticEmbeddingDataset(synthetic)` — cada item já é `(embedding, label)`, sem passar pelo backbone convolucional.
2. Combine com dados reais via `torch.utils.data.ConcatDataset`.
3. Use `EmbeddingClassifierWrapper(model)` para treinar/avaliar — isso exige que o modelo implemente `classify_from_embedding(features)` (já implementado em `Models/models.py:SVHNCustomCNN`), já que não há imagem para passar pelo backbone.

Essa arquitetura evita reimplementar um pipeline de treino paralelo só para dados sintéticos — o modelo simplesmente "pula" a etapa de extração de features quando já recebe o embedding pronto.

## Quando usar

Quando há poucas amostras minoritárias e replicar (oversampling puro) causaria overfitting severo nos mesmos pontos exatos — SMOTE gera variações plausíveis dentro da vizinhança da classe, ajudando o classificador a aprender uma fronteira de decisão mais suave ao redor da minoritária.

## Limitações

- Se a classe minoritária tiver poucas amostras (`k_neighbors >= len(grupo)`), a interpolação fica restrita a poucos vizinhos, podendo gerar pontos redundantes.
- Interpola linearmente no espaço de features — se esse espaço não for "suave" (pontos entre duas amostras reais não corresponderem a algo semanticamente válido), os sintéticos podem cair em regiões sem sentido. A qualidade do embedding usado (`model.get_embedding`) importa diretamente para a qualidade dos sintéticos.
- Não considera a posição da classe majoritária — pode gerar sintéticos minoritários muito perto da fronteira de decisão sem intenção (ver ADASYN, que resolve isso adaptativamente).
