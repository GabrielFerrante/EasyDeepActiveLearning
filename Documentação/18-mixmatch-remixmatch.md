# MixMatch / ReMixMatch (`train_model_with_mixmatch` / `train_model_with_remixmatch`)

**Categoria:** Sample scheduling aleatório + mixing
**Arquivo:** `Labeling_Strategies/mixmatch.py`
**Referências:**
- Berthelot, D., Carlini, N., Goodfellow, I., Papernot, N., Oliver, A. & Raffel, C. (2019). *"MixMatch: A Holistic Approach to Semi-Supervised Learning"*. NeurIPS.
- Berthelot, D., Carlini, N., Cubuk, E. D., Kurakin, A., Sohn, K., Zhang, H. & Raffel, C. (2019). *"ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring"*. ICLR 2020.

## MixMatch — ideia central

Uma abordagem "holística": em vez de uma única técnica (só pseudo-labeling, ou só consistência), MixMatch combina **label guessing** (média de predições sobre várias augmentations + sharpening) com **MixUp** entre exemplos rotulados e não rotulados, produzindo um único lote misto usado tanto para a loss supervisionada quanto para a não supervisionada.

### Algoritmo

1. **Label guessing** (Eqs. 6-7): para cada amostra não rotulada, gera `K` versões aumentadas (`MultiAugmentDataset`), passa todas pelo modelo, tira a **média** das probabilidades e aplica *sharpening* (`Labeling_Strategies/augmentation.py:sharpen`) — reduz a entropia da distribuição, aproximando-a de um one-hot.
2. Concatena `(x̂_b, p_b)` rotulado com `(û_{b,k}, q_b)` não rotulado (Eqs. 10-13), embaralha tudo, e aplica **MixUp** (`augmentation.py:mixup`) entre cada item original e um par aleatório do conjunto combinado.
3. `L_X` = cross-entropy no lote misto rotulado; `L_U` = MSE no lote misto não rotulado (Eqs. 3-4) — MSE em vez de cross-entropy porque o alvo `q_b` é soft, e o paper mostra que MSE é mais robusto a pseudo-rótulos incorretos nesse caso.
4. `L = L_X + λ_u·L_U` (Eq. 5) — `λ_u` tipicamente **muito maior que 1** no MixMatch (paper usa até 75), já que `L_U` (MSE) tem escala bem menor que cross-entropy.

### Assinatura

```python
train_model_with_mixmatch(model, labeled_loader, unlabeled_loader, test_loader, task,
                           optimizer, device, epochs, num_classes,
                           sharpening_temperature=0.5, alpha=0.75, lambda_u=75.0,
                           lambda_u_rampup_steps=None, writer=None, log_prefix="") -> (loss, test_metric)
```

`unlabeled_loader` precisa ser construído sobre `MultiAugmentDataset` (batch[0] é uma lista de `K` tensores). `labeled_loader`: rótulos como índice de classe (convertidos para one-hot internamente).

## ReMixMatch — o que muda

ReMixMatch reaproveita o núcleo do MixMatch (`_mixmatch_losses`) e acrescenta dois componentes:

- **Augmentation Anchoring**: em vez de tirar a média sobre `K` augmentations independentes para o label guessing, usa uma única augmentation **fraca** como "âncora" — mais estável, já que uma augmentation fraca não distorce o suficiente para corromper a predição usada como alvo. `unlabeled_loader` ainda precisa prover `K` versões por amostra; a primeira é tratada como âncora fraca, as demais como augmentations fortes usadas no MixUp.
- **Distribution Alignment**: a distribuição guessada `q_b` é re-escalada pela razão entre a distribuição de classes do conjunto **rotulado** (fixa, atualizada por EMA simples sobre os batches vistos) e a média móvel da distribuição de classes **prevista pelo modelo** no não rotulado (atualizada com EMA de fator `align_momentum`) — corrige o viés do modelo em favor das classes que ele já prevê mais, encorajando a distribuição marginal das predições no não rotulado a bater com a do rotulado.

### Assinatura

```python
train_model_with_remixmatch(model, labeled_loader, unlabeled_loader, test_loader, task,
                             optimizer, device, epochs, num_classes,
                             sharpening_temperature=0.5, alpha=0.75, lambda_u=1.5,
                             lambda_u_rampup_steps=None, align_momentum=0.999,
                             writer=None, log_prefix="") -> (loss, test_metric)
```

**Escopo excluído (documentado no código):** a loss auxiliar de *rotation prediction* (self-supervisão) do paper ReMixMatch fica **fora** desta implementação — é uma técnica de aprendizado de representação ortogonal a pseudo-labeling (mesmo raciocínio usado para excluir a loss de metric learning do Dynamic Curriculum Learning em `Balance_Strategies`).

## Requisito de task (guard adicionado)

Ambas as funções levantam `AttributeError` cedo se `not hasattr(task, "criterion")` — embora nenhuma das duas chame `task.criterion` diretamente (`L_X`/`L_U` são calculadas internamente, via cross-entropy/MSE hardcoded), o guard existe porque `_evaluate` chama `task.compute_metric`, que assume implicitamente uma task de classificação de rótulo único (one-hot) — uma task como `DetectionTask` quebraria de forma obscura dentro da avaliação, sem esse guard.

## Quando usar

Quando há orçamento para K augmentations por amostra não rotulada (mais caro computacionalmente que o par único fraco/forte de FixMatch/UDA) e o benefício do MixUp (misturar exemplos, suavizando a fronteira de decisão) é desejado. ReMixMatch tende a superar MixMatch quando o modelo desenvolve viés de classe visível nas predições sobre o não rotulado — a Distribution Alignment corrige isso diretamente.

## Limitações

- `λ_u` do MixMatch (até 75) é sensível à escala relativa de `L_X`/`L_U` — vale a pena monitorar as duas separadamente (o loop já reporta ambas via `progress_bar.set_postfix`).
- ReMixMatch sem a loss de rotation prediction não é 100% fiel ao paper original — só o núcleo de balanceamento/consistência foi implementado.
- Ambas assumem rótulo único, com `num_classes` fixo conhecido de antemão (usado para o one-hot dos rótulos reais).
