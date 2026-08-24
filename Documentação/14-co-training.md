# Co-Training (`co_training_strategy`)

**Categoria:** Multi-model (duas visões)
**Arquivo:** `Labeling_Strategies/labeling_strategies.py`
**Referência:** Blum, A. & Mitchell, T. (1998). *"Combining Labeled and Unlabeled Data with Co-Training"*. COLT.

## Ideia central

Dois classificadores, treinados sobre **duas visões diferentes** dos mesmos dados, rotulam um para o outro: cada um gera pseudo-rótulos, para as amostras em que está confiante, que servem de treino para o **outro** classificador na próxima rodada. A suposição teórica original é que as duas visões são condicionalmente independentes dado o rótulo verdadeiro — isso reduz a chance de os dois classificadores cometerem o mesmo erro ao mesmo tempo, então a confiança de um é um sinal relativamente confiável para o outro, mesmo sem rótulo real nenhum envolvido.

## Adaptação para imagens

Imagens não têm uma divisão natural de features em "duas visões" (diferente do exemplo clássico do paper — página web dividida em texto do link e texto da página). A adaptação usual, seguida por esta implementação, é tratar duas arquiteturas/inicializações diferentes, ou dois pipelines de augmentation diferentes, como as duas "visões": daí `model_a`/`model_b` e `unlabeled_loader_a`/`unlabeled_loader_b` serem parâmetros independentes — podem ser o mesmo dataset com augmentations diferentes, ou visões genuinamente distintas (ex.: dois crops, ou duas modalidades).

## Algoritmo

1. Roda `model_a` sobre `unlabeled_loader_a` e `model_b` sobre `unlabeled_loader_b`, obtendo confiança e predição por amostra.
2. Para toda amostra onde `model_a` está confiante (`>= confidence_threshold`): sua predição vira pseudo-rótulo **para treinar `model_b`**.
3. Para toda amostra onde `model_b` está confiante: sua predição vira pseudo-rótulo **para treinar `model_a`**.

Ou seja, cada modelo nunca "aprende com a própria opinião" — sempre recebe rótulos gerados pelo parceiro.

## Assinatura

```python
co_training_strategy(model_a, model_b, device, unlabeled_loader_a, unlabeled_loader_b,
                      unlabeled_indices, confidence_threshold=0.95, **kwargs) -> (result_a, result_b)
```

- `unlabeled_loader_a`/`unlabeled_loader_b`: ambos devem iterar na **mesma ordem** de `unlabeled_indices` (mesmo que apliquem augmentations diferentes à mesma imagem base).
- Retorno: `(pseudo_labels_para_a, pseudo_labels_para_b)` — cada um no formato `(selected_indices, labels)` de `pseudo_labeling_strategy`. `pseudo_labels_para_a` vem das predições confiantes de `model_b` (para treinar `a` com o que `b` rotulou), e vice-versa.

## Quando usar

Quando é viável treinar/manter dois modelos em paralelo com fontes de "visão" genuinamente diferentes (arquiteturas distintas, augmentations fortemente diferentes, ou modalidades de entrada distintas) — o ganho do método depende diretamente de quão pouco correlacionados são os erros dos dois classificadores. Se as duas "visões" forem essencialmente a mesma coisa (mesma arquitetura, augmentation trivialmente diferente), o método se aproxima de um pseudo-labeling clássico duplicado, sem o benefício teórico da independência condicional.

## Limitações

- A suposição de independência condicional das duas visões raramente é estritamente verdadeira em visão computacional (ao contrário do cenário original de features de texto) — o ganho prático depende de quão "diferentes" as duas visões conseguem ser na prática.
- Sem nenhum mecanismo de balanceamento de classe embutido (diferente de `pseudo_labeling_strategy`/`flexmatch_strategy`, que recebem `balance_fn`) — se necessário, aplique uma estratégia de `Balance_Strategies` manualmente sobre o resultado.
- Threshold único e fixo para as duas direções (mesmo problema do pseudo-labeling clássico, sem a adaptação por classe do FlexMatch).
