# Pseudo-Label Clássico (`pseudo_labeling_strategy`)

**Categoria:** Semi-supervisionado, PL original
**Arquivo:** `Labeling_Strategies/labeling_strategies.py`
**Referência:** Lee, D.-H. (2013). *"Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks"*. ICML Workshop on Challenges in Representation Learning.

## Ideia central

A forma mais simples de semi-supervisão: usa a própria predição do modelo como rótulo para amostras não rotuladas em que ele já está suficientemente confiante. Não há oráculo, não há segundo modelo, não há augmentation especial — só um threshold de confiança sobre o `argmax` do softmax. É a base histórica sobre a qual quase todas as técnicas mais recentes da biblioteca (FlexMatch, Noisy Student, FixMatch, UDA, MPL...) constroem variações.

**Nota sobre a citação (Lee, 2013)**: o Pseudo-Label original do paper **não** usa um threshold de confiança por amostra. A formulação original é uma única loss `L = L_sup + α(t)·L_unsup` aplicada sobre **todas** as amostras não rotuladas a cada época, com `α(t)` crescendo linearmente por partes ao longo do treino (0 numa fase inicial, depois sobe até um valor final `α_f`) — quanto mais o treino avança, mais peso o sinal não supervisionado ganha. O esquema de "threshold fixo" implementado aqui — e chamado de "pseudo-labeling clássico" pela maior parte da literatura posterior (FixMatch, UDA, FlexMatch todos herdam esse nome) — é essa variante popularizada depois, não uma leitura literal da equação do Lee 2013 original.

## Algoritmo

1. Roda o modelo (`model.eval()`) sobre todo o `unlabeled_loader`.
2. Para cada amostra, calcula `confidence = max(softmax(outputs))` e `pred = argmax(softmax(outputs))`.
3. Mantém como candidato toda amostra com `confidence >= confidence_threshold` (padrão `0.95`, valor típico da literatura).
4. Passa os candidatos por uma `balance_fn` (ver `Documentação/01-balanceamento-por-classe.md`) para evitar que a classe que o modelo já acerta mais — logo, a mais confiante — domine o conjunto pseudo-rotulado a cada ciclo, reforçando esse viés.

Generaliza para saídas multi-posição (ex.: `[Batch, N_posições, C]`, como o `SVHNCustomCNN`): a confiança por amostra é a média das confianças de cada posição.

## Assinatura

```python
pseudo_labeling_strategy(model, device, unlabeled_loader, unlabeled_indices, balance_fn,
                          confidence_threshold=0.95, max_per_class=None, **kwargs) -> (indices, labels)
```

- `unlabeled_loader`: `DataLoader` com `shuffle=False`, iterando na mesma ordem de `unlabeled_indices` (a posição no batch precisa corresponder ao índice real do dataset).
- `balance_fn`: `callable(candidates, max_per_class) -> candidatos balanceados` — qualquer estratégia de `Balance_Strategies/balance_strategies.py` que aceite 3-tuplas `(score, index, label)`.
- Retorno: `(selected_indices, pseudo_labels)` — índices originais do dataset aceitos e os rótulos previstos correspondentes, já balanceados.

## Uso na biblioteca — `PseudoLabeledDataset`

O resultado desta função normalmente alimenta `Labeling_Strategies.labeling_strategies.PseudoLabeledDataset(base_dataset, indices, pseudo_labels)`, que devolve, para cada índice, a imagem original combinada com o pseudo-rótulo **no lugar do rótulo real** — o rótulo verdadeiro do dataset base nunca é lido nem sobrescrito. Preserva os campos extras do item original (`(item[0], pseudo_label) + tuple(item[2:])`), para ter a mesma aridade do dataset rotulado de verdade — os dois são combinados num único `DataLoader` via `ConcatDataset`, e misturar tuplas de tamanhos diferentes no mesmo batch quebraria o `collate` do PyTorch.

**Princípio de design da biblioteca**: pseudo-rotulação é usada **apenas** para dados nunca rotulados por um oráculo — nunca substitui um rótulo real já existente.

## Qual o papel do pseudo-labeling quando já existe rótulo verdadeiro?

Numa Active Learning clássica (`run_active_learning_cycle`), a query strategy escolhe um pequeno orçamento de amostras para o oráculo rotular a cada ciclo — o restante do pool fica sem rótulo real até um ciclo futuro. `pseudo_labeling_strategy` pode atuar **sobre esse restante**, gerando sinal de treino extra (transiente, recalculado a cada ciclo) sem gastar orçamento de oráculo — a mesma lógica do CEAL (*Cost-Effective Active Learning*, Wang et al., 2016). Em `run_self_labeling_cycle`, o papel é ainda mais central: não existe oráculo real disponível, e a própria pseudo-rotulação atua como "oráculo" (com persistência permanente) para as amostras que a query escolheu.

## Quando usar

Como primeira estratégia de labeling em qualquer pipeline — é o baseline mais simples e barato. Vale trocar por FlexMatch quando o modelo aprende as classes em ritmos muito diferentes (um único `confidence_threshold` fixo favorece classes "fáceis"), ou por Noisy Student/FixMatch/UDA quando há orçamento para um loop de treino mais sofisticado com augmentation e/ou ruído.

## Limitações

- Threshold fixo e único para todas as classes — não se adapta ao ritmo de aprendizado por classe (ver FlexMatch).
- Confirmation bias: erros confiantes do modelo se reforçam ciclo após ciclo, já que o próprio modelo é a fonte do "rótulo".
- Sem nenhuma regularização por ruído/consistência — puramente confiança estática, ao contrário de FixMatch/UDA/VAT.
