# Noisy Student (`noisy_student_pseudo_label`)

**Categoria:** Self-training com ruído
**Arquivo:** `Labeling_Strategies/labeling_strategies.py`
**Referência:** Xie, Q., Luong, M.-T., Hovy, E. & Le, Q. V. (2020). *"Self-training with Noisy Student improves ImageNet classification"*. CVPR.

## Ideia central

Um ciclo iterativo de self-training onde um modelo **teacher** (já treinado, com dados rotulados) gera pseudo-rótulos para o pool não rotulado; um modelo **student** — igual ou **maior** que o teacher — é então treinado sobre rotulado + pseudo-rotulado, mas com **ruído injetado deliberadamente**: augmentation forte na entrada (ex.: RandAugment) e ruído no próprio modelo (dropout, stochastic depth). O processo se repete: o student vira o teacher da próxima iteração. A intuição central é que o ruído força o student a fazer mais do que apenas "imitar" o teacher — ele precisa generalizar o suficiente para reconstruir a predição correta mesmo sob perturbação, o que evita que a qualidade degrade ciclo após ciclo (ao contrário do self-training ingênuo).

## O que esta função implementa

Esta função é **apenas o passo de seleção** (geração dos pseudo-rótulos pelo teacher) — a parte "Noisy" do método está em **como o student é treinado depois**, não em como os candidatos são escolhidos. Configure o treino do student separadamente com augmentation forte (`Labeling_Strategies/augmentation.py`) e, se quiser fidelidade completa ao paper, ruído de modelo (dropout/stochastic depth na arquitetura). Repita o processo (student → teacher da próxima rodada) para reproduzir o algoritmo iterativo completo.

## Algoritmo

1. O teacher (`model`, já treinado) prediz sobre todo o `unlabeled_loader`.
2. Mantém como candidato toda amostra com confiança `>= confidence_threshold` — o paper usa **0.3**, bem mais permissivo que o `0.95` do pseudo-labeling clássico, já que o método confia na regularização por ruído do student para compensar rótulos ocasionalmente errados.
3. Agrupa candidatos por classe prevista.
4. Se `max_per_class` for informado, cada classe é cortada/**duplicada aleatoriamente** até esse tamanho — reproduz o comportamento do paper original, que limita a 130 mil exemplos por classe e duplica aleatoriamente classes com poucas imagens até atingir a cota, para manter o treino do student balanceado.

## Assinatura

```python
noisy_student_pseudo_label(model, device, unlabeled_loader, unlabeled_indices,
                            confidence_threshold=0.3, max_per_class=None, soft=False, **kwargs) -> (indices, labels)
```

- `confidence_threshold`: padrão `0.3` (bem mais baixo que o pseudo-labeling clássico — ver acima).
- `max_per_class`: se informado, ativa o corte/duplicação por classe descrito acima; se não, usa a contagem da maior classe como alvo implícito.
- `soft`: se `True`, devolve a distribuição de probabilidade inteira como "pseudo-rótulo" em vez do `argmax` — o paper reporta desempenho parecido entre hard/soft, com soft levemente melhor para dados fora do domínio de treino.
- Retorno: `(selected_indices, pseudo_labels)`, igual a `pseudo_labeling_strategy`.

## Quando usar

Quando há recursos para treinar um modelo maior que o atual e rodar múltiplas iterações teacher→student — é a técnica mais cara operacionalmente da biblioteca em termos de orquestração externa (não é uma chamada única, é um processo iterativo com re-treino completo a cada rodada). Compensa em datasets grandes o suficiente para o pool não rotulado fornecer sinal substancial mesmo com um threshold permissivo.

## Limitações

- Só cobre a seleção; a parte que realmente define o método (ruído no student) precisa ser montada manualmente pelo chamador.
- Threshold permissivo (`0.3`) sem nenhum mecanismo de correção (ao contrário de FlexMatch) — depende inteiramente do ruído no treino do student para não degradar com pseudo-rótulos errados.
- Duplicação aleatória para balancear classes pequenas (quando `max_per_class` é informado) pode reforçar erros específicos se o teacher estiver sistematicamente errado numa classe rara.
