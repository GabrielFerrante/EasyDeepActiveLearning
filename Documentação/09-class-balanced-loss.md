# Class-Balanced Loss (`class_balanced_weights`)

**Categoria:** Feature representation
**Arquivo:** `Balance_Strategies/losses.py`
**Referência:** Cui, Y., Jia, M., Lin, T.-Y., Song, Y. & Belongie, S. (2019). *"Class-Balanced Loss Based on Effective Number of Samples"*. CVPR.

## Ideia central

O cost-sensitive learning clássico pondera cada classe por `1/n_c` (inverse frequency). O paper de Cui et al. argumenta que isso não é exatamente correto: amostras adicionais de uma classe já bem representada tendem a se **sobrepor** (redundância de informação) — a segunda foto de um cachorro não dobra a informação que o modelo tem sobre "cachorro" do mesmo jeito que a primeira. Por isso, propõem ponderar pelo **"número efetivo de amostras"**, uma quantidade que cresce sub-linearmente com `n_c` em vez de linearmente.

## Fórmula

```
E_{n_c} = (1 - beta^{n_c}) / (1 - beta)          (número efetivo de amostras da classe c)
w_c = (1 - beta) / (1 - beta^{n_c}) = 1 / E_{n_c}
```

O comportamento de `beta ∈ [0, 1)` controla o quão agressivamente classes com muitas amostras são sub-ponderadas:

- `beta = 0` → `E_{n_c} = 1` para toda classe → sem reponderação nenhuma (`w_c = 1`).
- `beta → 1` → `E_{n_c} → n_c` → se aproxima de `1/n_c`, ou seja, do inverse frequency clássico.

O paper reporta bons resultados com `beta ∈ {0.9, 0.99, 0.999, 0.9999}`, dependendo do quão desbalanceado e do quão grande é o dataset.

## Assinatura

```python
class_balanced_weights(class_counts, beta=0.9999) -> torch.Tensor
```

- `class_counts`: array/lista com o número de amostras de cada classe (`n_c`).
- `beta`: fator de "sobreposição" (ver acima).
- Retorno: tensor de pesos, normalizados para média 1 (mesma convenção da implementação oficial dos autores) — pronto para usar como `weight` de `nn.CrossEntropyLoss` **ou** como `alpha` de `FocalLoss` (o próprio paper original combina os dois: Class-Balanced **Focal** Loss).

## Como usar na biblioteca

```python
weights = class_balanced_weights(class_counts=[5000, 3000, 200, 50], beta=0.9999)
task = ClassificationTask(criterion=FocalLoss(gamma=2.0, alpha=weights))
```

## Quando usar

Quando `compute_class_weights(scheme="inverse_frequency")` está super-corrigindo — dando peso desproporcional a classes minoritárias muito pequenas — Class-Balanced Loss costuma ser uma alternativa mais estável, porque a curva de `1/E_{n_c}` cresce mais devagar que `1/n_c` para valores pequenos de `n_c`, mesmo com `beta` alto.

## Limitações

- Escolher `beta` é um hiperparâmetro extra a ajustar — o paper sugere uma faixa, mas o valor ótimo varia por dataset.
- Como no cost-sensitive learning tradicional, pondera por classe inteira, não por instância — não distingue exemplos fáceis de difíceis dentro da mesma classe (para isso, combine com `FocalLoss` via `alpha`, como no exemplo acima).
