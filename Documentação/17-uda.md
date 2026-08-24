# UDA — Unsupervised Data Augmentation (`train_model_with_uda`)

**Categoria:** Consistency regularization (+ TSA)
**Arquivo:** `Labeling_Strategies/uda.py`
**Referência:** Xie, Q., Dai, Z., Hovy, E., Luong, M.-T. & Le, Q. V. (2019/2020). *"Unsupervised Data Augmentation for Consistency Training"*. NeurIPS 2020.

## Ideia central

Muito parecido com FixMatch (mesmo padrão de augmentation fraca/forte, mesma estrutura de loop), mas com duas diferenças centrais: (1) a loss de consistência compara contra uma distribuição-alvo **suavizada** (soft label, via softmax com temperatura) em vez de um pseudo-rótulo hard (`argmax`); (2) introduz o **Training Signal Annealing (TSA)**, um mecanismo que mascara exemplos **rotulados** conforme o treino avança, para evitar que o sinal supervisionado (poucos exemplos) sature/overfite antes do sinal não supervisionado (muitos exemplos) ter chance de contribuir.

## Fórmulas

Objetivo geral (Eq. 1 do paper):

```
J(θ) = E[-log p_θ(y*|x1)] + λ·E_x2[ E_x̂~q(x̂|x2)[ CE( p_θ̃(y|x2) , p_θ(y|x̂) ) ] ]
```

onde `θ̃` é uma cópia **fixa** dos parâmetros atuais (sem gradiente) — o alvo da consistência nunca propaga gradiente de volta.

- **Confidence-based masking**: a consistency loss só é computada nos exemplos não rotulados cuja confiança em `p_θ̃(y|x2)` (predição na versão **fracamente** aumentada, sem gradiente) ultrapassa `confidence_threshold` (`0.8` no paper, para CIFAR-10/SVHN).
- **Sharpening**: `p_θ̃(y|x2)` é suavizada com temperatura (`sharpening_temperature=0.4` no paper) antes de virar alvo da cross-entropy — reforça minimização de entropia implícita.
- **TSA** (`tsa_threshold`, `use_tsa=True`): threshold crescente `η_t` que mascara exemplos rotulados cuja confiança na classe certa já ultrapassa `η_t`:
  ```
  η_t = α_t·(1 - 1/K) + 1/K,   K = num_classes
  ```
  `α_t` sobe de 0 a 1 conforme `progress = step/total_steps`, com três formatos de curva (`tsa_schedule`): `"linear"` (constante), `"log"` (sobe rápido no início, satura no fim), `"exp"` (sobe devagar no início, rápido no fim).

## Assinatura

```python
train_model_with_uda(model, labeled_loader, unlabeled_loader, test_loader, task,
                      optimizer, device, epochs, num_classes,
                      confidence_threshold=0.8, sharpening_temperature=0.4, lambda_u=1.0,
                      use_tsa=True, tsa_schedule="linear",
                      writer=None, log_prefix="") -> (loss, test_metric)
```

- `unlabeled_loader`: precisa ser construído sobre `WeakStrongAugmentDataset` (mesma convenção do FixMatch).
- `num_classes`: necessário para o TSA (define o threshold inicial `1/num_classes`).
- Requer `task` com atributo `.criterion`.
- Generaliza para saída multi-posição: as losses supervisionada e de consistência achatam dimensões extras antes do cross-entropy manual (`_reduce_sample_dims`), diferente do FixMatch, que usa `task.criterion` diretamente.

## Quando usar

Quando o conjunto rotulado é muito pequeno em relação ao não rotulado — o TSA foi desenhado exatamente para esse cenário, evitando que o modelo memorize rapidamente os poucos exemplos rotulados antes do sinal não supervisionado (muito mais volumoso) conseguir contribuir. Se o conjunto rotulado já for razoavelmente grande, o ganho do TSA sobre o FixMatch tende a ser menor.

## Limitações

- Mais hiperparâmetros que FixMatch (`sharpening_temperature`, `tsa_schedule`, além do threshold) — mais superfície para ajuste fino.
- A suposição do TSA (sinal rotulado convergindo "rápido demais") só se aplica quando o rotulado é escasso; em cenários com bastante rótulo disponível, `use_tsa=False` pode ser preferível.
