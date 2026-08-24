# Documentação

Um arquivo por técnica implementada na biblioteca, cada um com a ideia central, o algoritmo/fórmulas, a assinatura da função/classe, quando usar e as limitações conhecidas. Acompanha o roadmap de métodos em [`readme.md`](../readme.md#métodos-de-referência-roadmap).

## Balance Strategies

### Re-labeling

- [01 — Balanceamento por classe/quota](01-balanceamento-por-classe.md) (`class_balance_strategy`)

### Data re-balancing (sem features)

- [02 — Random Oversampling / Undersampling](02-random-oversampling-undersampling.md) (`random_oversample_strategy`, `random_undersample_strategy`)

### Data re-balancing (undersampling por vizinhança)

- [03 — Tomek Links](03-tomek-links.md) (`tomek_links_strategy`)
- [04 — NearMiss](04-near-miss.md) (`near_miss_strategy`)

### Data re-balancing (geração sintética)

- [05 — SMOTE](05-smote.md) (`smote_oversample`)
- [06 — ADASYN](06-adasyn.md) (`adasyn_oversample`)

### Feature representation (loss-reweighting)

- [07 — Cost-Sensitive Learning / Class Weights](07-cost-sensitive-class-weights.md) (`compute_class_weights`)
- [08 — Focal Loss](08-focal-loss.md) (`FocalLoss`)
- [09 — Class-Balanced Loss](09-class-balanced-loss.md) (`class_balanced_weights`)
- [10 — Dynamic Curriculum Learning](10-dynamic-curriculum-learning.md) (`DynamicCurriculumLoss`)

## Labeling Strategies

### Seleção (funções, sem loop de treino próprio)

- [11 — Pseudo-Label Clássico](11-pseudo-label-classico.md) (`pseudo_labeling_strategy`, `PseudoLabeledDataset`)
- [12 — FlexMatch](12-flexmatch.md) (`flexmatch_strategy`)
- [13 — Noisy Student](13-noisy-student.md) (`noisy_student_pseudo_label`)
- [14 — Co-Training](14-co-training.md) (`co_training_strategy`)
- [15 — Tri-Training](15-tri-training.md) (`tri_training_strategy`)

### Treino (loop próprio, substitui `Training/train.py`; exigem `task.criterion`)

- [16 — FixMatch](16-fixmatch.md) (`train_model_with_fixmatch`)
- [17 — UDA](17-uda.md) (`train_model_with_uda`)
- [18 — MixMatch / ReMixMatch](18-mixmatch-remixmatch.md) (`train_model_with_mixmatch`, `train_model_with_remixmatch`)
- [19 — Mean Teacher](19-mean-teacher.md) (`train_model_with_mean_teacher`)
- [20 — VAT](20-vat.md) (`train_model_with_vat`)
- [21 — MPL (Meta Pseudo Labels)](21-mpl.md) (`train_model_with_mpl`)
