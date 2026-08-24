# Tri-Training (`tri_training_strategy`)

**Categoria:** Multi-model (três classificadores)
**Arquivo:** `Labeling_Strategies/labeling_strategies.py`
**Referência:** Zhou, Z.-H. & Li, M. (2005). *"Tri-training: Exploiting Unlabeled Data Using Three Classifiers"*. IEEE Transactions on Knowledge and Data Engineering.

## Ideia central

Uma extensão de Co-Training para **três** classificadores, que elimina a necessidade de "visões" independentes explícitas: cada classificador `h_i` é (idealmente) treinado numa amostra bootstrap diferente do conjunto rotulado. A cada rodada, uma amostra não rotulada entra no conjunto de treino de `h_i` **se os outros dois classificadores (`h_j`, `h_k`) concordam** na predição — a concordância entre dois modelos independentes funciona como "voto de confiança", sem precisar de nenhum threshold de probabilidade.

## Critério de aceitação (fiel ao paper original)

Aceitar cegamente tudo que `h_j`/`h_k` concordam pode introduzir ruído demais se a taxa de erro do par for alta. O paper propõe um critério mais cauteloso:

1. Estima `e_i`, a **taxa de erro combinada** do par `(h_j, h_k)` sobre um conjunto rotulado (`labeled_loader`): a fração das vezes que os dois concordam **e** erram (comparado ao rótulo verdadeiro).
2. Só aceita atualizar `h_i` nesta rodada se `e_i` for **menor** que o `e_i` aceito na rodada **anterior** (`prev_error`) — a taxa de erro do par precisa estar melhorando.
3. Se o novo conjunto de candidatos (amostras onde `h_j`/`h_k` concordam) for grande demais para o `e_i` atual, subamostra para manter:
   ```
   e_i · |L_i| < prev_e_i · |prev_L_i|
   ```
   ou seja, o "orçamento" esperado de rótulos errados introduzidos no treino de `h_i` não pode crescer de uma rodada para a outra — é essa invariante que fundamenta teoricamente a convergência do método no paper original.

## Assinatura

```python
tri_training_strategy(models, device, unlabeled_loader, unlabeled_indices,
                       labeled_loader, prev_errors=None, prev_sizes=None,
                       random_state=42, **kwargs) -> [(indices_1, labels_1, error_1, size_1), ...]
```

- `models`: lista de exatamente 3 modelos `[h1, h2, h3]`.
- `prev_errors`/`prev_sizes`: listas de 3 valores — o `e_i` e `|L_i|` **aceitos na rodada anterior**; `None` na primeira rodada (nesse caso, tudo que passar no critério de concordância é aceito, sem checagem de melhora de erro).
- `labeled_loader`: usado **apenas** para estimar `e_i` (precisa de rótulo verdadeiro — não entra no treino direto desta função).
- Retorno: lista de 3 tuplas `(selected_indices, pseudo_labels, error_i, size_i)`, uma por classificador. `error_i`/`size_i` devem ser passados como `prev_errors[i]`/`prev_sizes[i]` na **próxima** chamada, para manter o critério de aceitação consistente entre rodadas — o estado do método vive fora da função, no código que orquestra o loop de rodadas.

## Quando usar

Quando não há uma divisão natural de "visões" para Co-Training, mas é viável manter três modelos (idealmente com inicializações/arquiteturas ou amostras de bootstrap diferentes) — o critério de aceitação baseado em taxa de erro estimada é mais conservador e teoricamente fundamentado que um threshold de confiança fixo, o preço sendo a necessidade de manter um `labeled_loader` disponível a cada rodada só para estimar `e_i`.

## Limitações

- Requer chamar em loop, mantendo `prev_errors`/`prev_sizes` entre chamadas — não é uma função de seleção "stateless" como as demais.
- A estimativa de `e_i` depende do tamanho do `labeled_loader`: com poucos exemplos rotulados, a estimativa de taxa de erro do par fica ruidosa, tornando o critério de aceitação instável.
- Assume rótulo único (a comparação `pred_j != targets` usa `argmax` direto) — não generaliza para saídas multi-posição sem adaptação.
