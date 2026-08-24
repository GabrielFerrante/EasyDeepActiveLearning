# Focal Loss (`FocalLoss`)

**Categoria:** Feature representation (nível de instância)
**Arquivo:** `Balance_Strategies/losses.py`
**Referência:** Lin, T.-Y., Goyal, P., Girshick, R., He, K. & Dollár, P. (2017). *"Focal Loss for Dense Object Detection"*. ICCV.

## Ideia central

Diferente do cost-sensitive learning (que pondera por **classe**), Focal Loss pondera por **instância**, com base em quão confiante o modelo já está no exemplo:

```
FL(p_t) = -alpha_t · (1 - p_t)^gamma · log(p_t)
```

onde `p_t` é a probabilidade que o modelo atribuiu à classe verdadeira. O fator `(1-p_t)^gamma` reduz a contribuição de exemplos já bem classificados (`p_t` alto → `(1-p_t)^gamma` pequeno), concentrando o gradiente do treino nos exemplos **difíceis** (`p_t` baixo). Como amostras de classes minoritárias tendem, em média, a ser mais difíceis para o modelo (ele viu menos exemplos delas), a Focal Loss ajuda com desbalanceamento mesmo sem nenhum peso de classe explícito (`alpha=None`).

O termo `alpha_t` opcional é um peso por classe adicional (pode vir de `compute_class_weights`/`class_balanced_weights`), combinando a modulação por dificuldade com cost-sensitive learning tradicional.

## Parâmetros

- `gamma` (padrão `2.0`, valor do paper): fator de modulação. `gamma=0` recupera exatamente a cross-entropy comum (sem nenhum efeito de foco).
- `alpha`: peso por classe, opcional — escalar ou tensor de tamanho `num_classes`.
- `ignore_index` (padrão `-100`): rótulos com esse valor são excluídos do cálculo da loss.

## Assinatura

```python
FocalLoss(gamma=2.0, alpha=None, ignore_index=-100)  # nn.Module
loss = focal(outputs, targets)   # outputs: [Batch, ..., C]   targets: [Batch, ...]
```

Suporta a mesma convenção de saída multi-posição do resto da biblioteca — achata `outputs`/`targets` internamente antes de calcular.

## Detalhe de implementação importante — evitar o double-weighting

`p_t` **precisa** vir de uma cross-entropy **sem** peso de classe. A implementação calcula primeiro `ce = F.nll_loss(log_probs, targets, reduction="none")` (sem `alpha`), depois `p_t = exp(-ce)`, e só então aplica `alpha_t` como um fator multiplicativo **separado**, depois do termo `(1-p_t)^gamma`:

```python
loss = ((1 - p_t) ** gamma) * ce
if alpha is not None:
    loss = alpha_t * loss
```

Se `alpha` entrasse dentro do `nll_loss` usado para derivar `p_t`, o resultado seria `exp(-alpha_t·ce) = p_t^alpha_t` em vez de `p_t` — corrompendo o significado de "probabilidade prevista para a classe verdadeira" e distorcendo o termo de modulação `(1-p_t)^gamma`. Um teste de regressão simples confirma a implementação: `FocalLoss(gamma=0, alpha=None)` bate exatamente com `nn.CrossEntropyLoss` padrão.

## Quando usar

Quando o desbalanceamento vem acompanhado de exemplos "fáceis" que dominam numericamente o treino (cenário clássico de detecção de objetos, onde a maioria das regiões de fundo são triviais de classificar) — Focal Loss deixa o treino focar no que realmente precisa de gradiente. Pode ser combinada com `alpha` vindo de `compute_class_weights`/`class_balanced_weights` quando, além do desbalanceamento de dificuldade, também há desbalanceamento de contagem entre classes.

## Limitações

- `gamma` alto demais pode fazer o modelo praticamente ignorar exemplos já razoavelmente bem classificados, atrasando a convergência fina.
- Não resolve, por si só, desbalanceamento de contagem entre classes — para isso, precisa do `alpha` explícito.
