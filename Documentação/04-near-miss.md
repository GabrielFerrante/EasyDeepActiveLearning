# NearMiss (`near_miss_strategy`)

**Categoria:** Data re-balancing (undersampling por vizinhança)
**Arquivo:** `Balance_Strategies/balance_strategies.py`
**Referência:** Mani, I. & Zhang, I. (2003). *"kNN Approach to Unbalanced Data Distributions: A Case Study Involving Information Extraction"*. Workshop on Learning from Imbalanced Datasets, ICML.

## Ideia central

Enquanto Random Undersampling remove amostras majoritárias ao acaso, NearMiss escolhe **quais** amostras majoritárias manter com base na distância até a classe minoritária, em três variantes com critérios diferentes de "informatividade":

- **Versão 1** — mantém as majoritárias mais próximas da minoritária (perto da fronteira de decisão). Foco em amostras difíceis/ambíguas.
- **Versão 2** — mantém as majoritárias com menor distância média aos vizinhos minoritários **mais distantes** (perto do "centro" da nuvem minoritária). Evita outliers ruidosos da fronteira, mais estável que a versão 1.
- **Versão 3** — algoritmo em **duas etapas**. Etapa 1: para cada amostra minoritária, mantém seus `n_neighbors` vizinhos majoritários mais próximos — garante que toda região da minoritária tenha vizinhança majoritária representada (útil quando a minoritária está espalhada em vários clusters), mas tipicamente forma um conjunto candidato bem maior que o tamanho final desejado. Etapa 2: dentro desse candidato, mantém só os `target_count` com **maior** distância média aos seus vizinhos minoritários mais próximos — descarta os membros mais "redundantes" do candidato, mantendo os mais representativos.

## Algoritmo

1. Separa candidatos em majoritária/minoritária (`_split_majority_minority`).
2. **v1**: para cada amostra majoritária, calcula a distância média aos `n_neighbors` vizinhos minoritários mais próximos; mantém as `target_count` majoritárias com **menor** distância média.
3. **v2**: para cada amostra majoritária, calcula a distância média aos `n_neighbors` vizinhos minoritários mais **distantes** (dentre todos os minoritários); mantém as `target_count` com menor distância média entre esses distantes.
4. **v3, etapa 1**: para cada amostra minoritária, busca seus `n_neighbors` vizinhos majoritários mais próximos; a união de todos esses vizinhos (sem repetição) forma o conjunto **candidato**.
5. **v3, etapa 2**: dentro do candidato, calcula a distância média de cada majoritária aos seus `n_neighbors` vizinhos minoritários mais próximos; mantém as `target_count` com **maior** distância média (oposto de v1 — aqui se quer as mais "seguras"/representativas do candidato, não as mais próximas da fronteira).

## Assinatura

```python
near_miss_strategy(candidates, target_count=None, version=1, n_neighbors=3) -> list
```

- `candidates`: lista de 4-tuplas `(score, index, label, features)` — precisa de embedding, como Tomek Links.
- `target_count`: tamanho final da classe majoritária, nas 3 versões (na v3, filtra o conjunto candidato da etapa 1); se não informado, iguala à contagem da minoritária.
- `version`: `1`, `2` ou `3` (ver acima).
- `n_neighbors`: quantos vizinhos considerar no cálculo de distância/seleção, usado em ambas as etapas da v3.
- Retorno: lista combinada (majoritária reduzida + minoritária inteira), 4-tupla.

## Quando usar

- **v1** quando o objetivo é focar o treino na fronteira de decisão (amostras difíceis).
- **v2** quando a v1 estiver capturando outliers ruidosos perto da minoritária (amostras mal rotuladas, por exemplo).
- **v3** quando a minoritária está espalhada em subgrupos distintos e é importante que cada subgrupo tenha contexto majoritário ao redor, não só uma região concentrada — e ainda assim se quer um tamanho final controlável via `target_count` (diferente de manter cegamente toda a união da etapa 1).

## Limitações

Assim como Tomek Links, assume um cenário majoritária-vs-resto e depende inteiramente da qualidade do espaço de embedding usado (`features`) — se o embedding não separa bem as classes, a noção de "proximidade" perde significado. Também é `O(n log n)` por causa da busca de vizinhos, repetida por versão (e, na v3, duas vezes — uma por etapa).
