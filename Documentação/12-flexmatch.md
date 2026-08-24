# FlexMatch (`flexmatch_strategy`)

**Categoria:** Curriculum Pseudo Labeling (threshold adaptativo por classe)
**Arquivo:** `Labeling_Strategies/labeling_strategies.py`
**Referência:** Zhang, B., Wang, Y., Hou, W., Wu, H., Wang, J., Okumura, M. & Shinozaki, T. (2021). *"FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling"*. NeurIPS.

## Ideia central

O pseudo-labeling clássico usa um único threshold de confiança (`tau`) para todas as classes — mas nem todas as classes são igualmente fáceis de aprender. Uma classe "difícil" pode nunca atingir `tau`, ficando permanentemente sem representação no conjunto pseudo-rotulado, mesmo que o modelo já tenha alguma noção dela. FlexMatch propõe um **threshold adaptativo por classe**: classes cujas amostras raramente atingem `tau` (aprendidas mais devagar) recebem um threshold **reduzido**, deixando mais amostras delas entrarem; classes já bem aprendidas mantêm o threshold próximo do original.

## Fórmulas

```
σ(c) = |{u : argmax p(y|u) = c  e  max p(y|u) > tau}|          (Eq. 5 — quantas amostras não rotuladas
                                                                  o modelo já classifica como c com confiança)
β(c) = σ(c) / max_c σ(c)                                        (Eq. 6, sem warm-up — "nível de aprendizado" de c)
T(c) = β(c) · tau                                                (Eq. 7 — threshold efetivo da classe c)
```

Com **warm-up** (`use_warmup=True`, recomendado — Eq. 11), o denominador de `β(c)` também conta as amostras ainda "não usadas" por nenhuma classe: `denom = max(max_c σ(c), N - Σ_c σ(c))`. Isso evita confiar demais na estimativa de `β(c)` nas primeiras iterações, quando as poucas amostras já confiantes podem estar concentradas por acaso numa única classe (o que inflaria `β` artificialmente para essa classe e zeraria as demais).

## Assinatura

```python
flexmatch_strategy(model, device, unlabeled_loader, unlabeled_indices, balance_fn,
                    num_classes, tau=0.95, use_warmup=True, max_per_class=None, **kwargs) -> (indices, labels)
```

- `num_classes`: necessário para calcular `σ(c)` para todo `c`.
- `tau`: threshold base (padrão `0.95`, mesmo valor da literatura de FixMatch/pseudo-labeling clássico).
- `use_warmup`: ativa o denominador com warm-up da Eq. 11 (recomendado).
- Retorno: `(selected_indices, pseudo_labels)`, igual a `pseudo_labeling_strategy`.

Recalculado do zero a cada chamada, a partir das predições atuais do modelo sobre **todo** o `unlabeled_loader` — equivalente a uma iteração `t` do Algorithm 1 do paper.

## Limitação de generalização (diferente de `pseudo_labeling_strategy`)

Assume classificação de **rótulo único** (saída `[Batch, num_classes]`) — o conceito de "threshold por classe" não generaliza diretamente para saídas multi-posição como a do exemplo SVHN (`[Batch, N_posições, C]`), já que `σ(c)`/`β(c)` são agregados por classe globalmente, não por posição.

## Quando usar

Quando classes têm dificuldade de aprendizado visivelmente desigual (algumas convergem rápido, outras ficam sempre abaixo do threshold fixo) — cenário comum em datasets com desbalanceamento natural ou classes visualmente parecidas entre si. Se as classes forem razoavelmente homogêneas em dificuldade, o ganho sobre o pseudo-labeling clássico costuma ser pequeno e não compensa o custo extra de recalcular `σ(c)` a cada chamada.

## Limitações

- Não funciona com saída multi-posição (ver acima).
- `σ(c)` é recalculado do zero a cada chamada (não há estado persistente entre ciclos, ao contrário da implementação original do paper, que mantém uma tabela de estimativas ao longo de todo o treino) — mais simples de integrar no ciclo de Active Learning da biblioteca, mas perde um pouco de suavidade temporal.
