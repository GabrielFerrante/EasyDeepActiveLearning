# FixMatch (`train_model_with_fixmatch`)

**Categoria:** Sample scheduling por métrica (threshold + augmentation fraca/forte)
**Arquivo:** `Labeling_Strategies/fixmatch.py`
**Referência:** Sohn, K., Berthelot, D., Li, C.-L., Zhang, Z., Carlini, N., Cubuk, E. D., Kurakin, A., Zhang, H. & Raffel, C. (2020). *"FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence"*. NeurIPS.

## Ideia central

Combina duas ideias que já existiam separadamente (pseudo-labeling e consistency regularization) numa receita simples e altamente eficaz: gera um pseudo-rótulo a partir da predição do modelo sobre uma versão **fracamente** aumentada da imagem (flip/shift) e, se a confiança ultrapassa um threshold, usa esse pseudo-rótulo como alvo para treinar o modelo a classificar corretamente a versão **fortemente** aumentada da mesma imagem (ex.: RandAugment). Diferente das estratégias de `labeling_strategies.py` (que decidem candidatos e devolvem um pseudo-rótulo estático, consumido depois por um loop de treino separado), FixMatch é uma **técnica de treino completa**: o pseudo-rótulo é recomputado a cada passo, a partir do modelo mais atual — por isso tem loop próprio, substituindo `Training/train.py`.

## Fórmulas

```
ℓ_s = (1/B) Σ H(y_b, p_m(y|α(x_b)))                                         — supervisionada (Eq. 2)
ℓ_u = (1/μB) Σ 1(max(q_b) ≥ τ) · H(q̂_b, p_m(y|A(u_b))),  q_b = p_m(y|α(u_b))  — pseudo-rótulo (Eq. 3-4)
loss = ℓ_s + λ_u · ℓ_u
```

onde `α` é augmentation fraca, `A` é forte, `q̂_b = argmax(q_b)` é o pseudo-rótulo hard (não soft), e `H` é cross-entropy. Sem ramp-up de `λ_u` — o paper mostra que não é necessário para FixMatch (ao contrário de UDA/Mean Teacher/MixMatch, que costumam usar).

## Assinatura

```python
train_model_with_fixmatch(model, labeled_loader, unlabeled_loader, test_loader, task,
                           optimizer, device, epochs, confidence_threshold=0.95, lambda_u=1.0,
                           writer=None, log_prefix="") -> (loss, test_metric)
```

- `unlabeled_loader`: precisa ser construído sobre `Labeling_Strategies.augmentation.WeakStrongAugmentDataset` — cada batch é `(weak_image, strong_image, ...)`.
- `labeled_loader`: `DataLoader` normal, com a **mesma augmentation fraca** aplicada (o paper usa flip+shift para os dois — rotulado e não rotulado).
- `confidence_threshold`: padrão `0.95`, mesmo valor do pseudo-labeling clássico.
- `lambda_u`: peso da loss não supervisionada (padrão `1.0`, sem ramp-up).
- Requer `task` com atributo `.criterion` (ex.: `ClassificationTask`) — levanta `AttributeError` cedo se ausente.
- Reporta `mask_rate` (fração de amostras não rotuladas que passaram do threshold) a cada época — útil para diagnosticar se o threshold está alto/baixo demais para o estágio atual do treino.

## Quando usar

É a técnica de treino semi-supervisionado mais simples e amplamente citada da biblioteca, com bom desempenho mesmo sem os componentes extras de UDA (TSA) ou MixMatch (mixup/distribution alignment) — boa opção padrão quando é viável montar um pipeline de augmentation fraca/forte. Requer um orçamento de treino maior que as estratégias de seleção pura de `labeling_strategies.py`, já que substitui o loop de treino inteiro.

## Limitações

- Assume rótulo único — a supervisão usa `task.criterion` diretamente, sem achatamento explícito de dimensões extras (diferente de UDA/VAT, que já reduzem dimensões antes do cross-entropy manual).
- Threshold fixo (não adaptativo por classe, ao contrário de FlexMatch) — classes difíceis podem contribuir pouco para `ℓ_u` durante boa parte do treino.
- Depende inteiramente da qualidade do par augmentation fraca/forte escolhido — augmentation forte fraca demais reduz o benefício da consistência; forte demais pode distorcer a imagem a ponto de o pseudo-rótulo deixar de fazer sentido.
