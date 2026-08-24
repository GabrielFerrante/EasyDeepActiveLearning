# VAT — Virtual Adversarial Training (`train_model_with_vat`)

**Categoria:** Consistency regularization (perturbação adversarial)
**Arquivo:** `Labeling_Strategies/vat.py`
**Referência:** Miyato, T., Maeda, S., Koyama, M. & Ishii, S. (2018). *"Virtual Adversarial Training: a Regularization Method for Supervised and Semi-Supervised Learning"*. IEEE TPAMI.

## Ideia central

Diferente de FixMatch/UDA (que usam augmentation de imagem como fonte de "ruído" para a consistência), VAT calcula explicitamente a **perturbação adversarial** — a direção no espaço de entrada que mais muda a predição do modelo — e treina o modelo para ser consistente mesmo sob essa perturbação. Como a perturbação é calculada a partir da predição atual do modelo (não do rótulo verdadeiro), o método funciona igualmente em dados rotulados **e** não rotulados, sem nunca gerar um pseudo-rótulo.

## Fórmulas

A "Local Distributional Smoothness" mede quão diferente a predição do modelo fica ao mover a entrada na direção adversarial:

```
LDS(x,θ) = D[p(y|x,θ̂), p(y|x+r_vadv,θ)]                    (Eq. 5)
```

`r_vadv` é aproximado via **power iteration** (Eqs. 9-15), sem calcular a Hessiana explicitamente:

```
d ← normalize(∇_r D[p(y|x,θ̂), p(y|x+r,θ̂)]|_{r=ξd}),   repetido power_iterations vezes
r_vadv ≈ ε · d
```

onde `θ̂` é o modelo atual tratado como **fixo** durante a busca (sem gradiente para os parâmetros — só para a direção `d`). O objetivo final de treino (Eq. 8):

```
loss = ℓ(D_l,θ) + α·R_vadv(D_l,D_ul,θ)
```

`ℓ` é a NLL supervisionada padrão; `R_vadv` é a média de `LDS(x,θ)` sobre rotulado **e** não rotulado. `α` e `ε` (`epsilon`) são os dois hiperparâmetros centrais do método — o paper reporta que, na prática, só `epsilon` costuma precisar de ajuste fino (`alpha=1` funciona bem na maioria dos casos).

## Detalhe de implementação importante — diferenciar em relação a `r`, não a `d`

`compute_vat_perturbation` faz `r = (xi * d).detach().requires_grad_()` e diferencia a divergência KL em relação a **`r`**, não a `d`. Diferenciar em relação a `d` introduziria um fator `xi` extra pela regra da cadeia (`d(x+xi·d)/d(d) = xi`), fazendo o gradiente perder precisão numérica para valores pequenos de `xi` (o padrão do paper, `1e-6`, é pequeno o suficiente para isso colapsar em float32). Essa era uma implementação incorreta encontrada e corrigida durante o desenvolvimento da biblioteca — a versão atual verificadamente aproxima `r_vadv.norm() ≈ epsilon`.

Nota prática sobre `xi`: como `xi` só serve para achar a **direção** (o resultado final é normalizado e reescalado por `epsilon`), seu valor exato não deveria importar matematicamente — mas `xi` extremamente pequeno pode deixar o gradiente da busca abaixo da precisão útil do float32, especialmente em modelos pouco treinados ou com entradas em escala grande. Se `r_vadv.norm()` sair muito menor que `epsilon`, aumente `xi` (ex.: `1e-2` a `1.0`).

O modelo é colocado em `eval()` durante a busca da perturbação (BatchNorm não deve atualizar estatísticas com entradas artificiais), e `model.train(training_mode)` é restaurado no `finally` — mesmo se a busca lançar uma exceção no meio.

## Assinatura

```python
train_model_with_vat(model, labeled_loader, unlabeled_loader, test_loader, task,
                      optimizer, device, epochs, epsilon=8.0, xi=1e-6, power_iterations=1, alpha=1.0,
                      writer=None, log_prefix="") -> (loss, test_metric)
```

- `epsilon`: magnitude da perturbação adversarial (o hiperparâmetro que mais importa ajustar).
- `xi`: tamanho do passo de diferença finita na power iteration (ver nota acima).
- `power_iterations`: repetições da power iteration (paper mostra que 1 já é suficiente na prática).
- `alpha`: peso de `R_vadv` na loss total.
- Requer `task` com atributo `.criterion`.

## Quando usar

Quando não há um pipeline de augmentation forte já pronto (FixMatch/UDA dependem dele) — VAT gera sua própria "perturbação mais informativa" diretamente a partir do modelo, sem exigir nenhuma escolha de augmentation. Também é a única técnica da biblioteca que funciona igualmente bem em dados **rotulados**, já que a perturbação não depende do rótulo verdadeiro.

## Limitações

- Sensível a `xi` (ver nota de precisão numérica acima) — vale a pena verificar `r_vadv.norm()` contra `epsilon` ao adaptar para um modelo/dataset novo.
- Cada passo de treino exige passadas forward extras para a busca da perturbação (uma por `power_iterations`, mais a passada final) — mais caro por passo que FixMatch/pseudo-labeling clássico.
- Não gera nenhum pseudo-rótulo explícito — não serve para os cenários da biblioteca que precisam de rótulos persistíveis (ex.: `run_self_labeling_cycle`), só para regularização durante o treino.
