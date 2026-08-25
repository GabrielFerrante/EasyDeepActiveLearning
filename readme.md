## Easy Deep Active Learning

Biblioteca em Python/PyTorch para Deep Active Learning em visão computacional. A ideia é ser agnóstica a:

- **Modelo**: qualquer `nn.Module` do PyTorch (o único contrato extra é `get_embedding(x)`, exigido só pelas estratégias de seleção baseadas em embedding).
- **Dataset**: você constrói seus próprios `DataLoader`s (dataset, transforms, batch size ficam por sua conta).
- **Tarefa**: Classificação, Segmentação, Detecção ou VLM contrastivo (estilo CLIP) — cada uma com sua própria forma de calcular loss, métrica e incerteza.
- **Estratégia de seleção (query)**: incerteza, densidade ou diversidade, escolhida por ciclo de Active Learning.
- **Estratégia de rotulação (labeling)**: como obter o rótulo de uma amostra não rotulada — oráculo (padrão) ou pseudo-labeling (o próprio modelo se rotula, quando confiante o suficiente).
- **Estratégia de balanceamento**: como evitar que uma classe domine uma seleção (hoje usada pela pseudo-labeling, mas pensada pra ser reaproveitável).

Um exemplo de ponta a ponta com o dataset SVHN está em [`Example_SVHN_ActiveLearning.ipynb`](Example_SVHN_ActiveLearning.ipynb), simulando um cenário sem oráculo real: a query escolhe candidatos incertos e o pseudo-labeling assume o papel de "oráculo" para eles.

### Estrutura do projeto

```
Data/                 dados e scripts de preparação de dados
Models/                modelos PyTorch (models.py traz um exemplo de CNN pra dígitos do SVHN)
Query_Strategies/     estratégias de seleção pro oráculo (querys_strategies.py + lpl.py/vaal.py/waal.py)
Labeling_Strategies/  rotulação sem oráculo: labeling_strategies.py + fixmatch/uda/mixmatch/mean_teacher/vat/mpl.py
Balance_Strategies/   balanceamento por classe (balance_strategies.py: resampling; losses.py: loss functions)
Training/              loop de treino, ciclo de Active Learning e as Tasks (Training/tasks.py)
Metrics/                métricas de avaliação por tarefa (task_metrics.py) e do processo de AL (al_metrics.py)
Utils/                 utilitários gerais
MethodsReferences/    artigos de referência (surveys) que embasam os métodos da lib
Documentação/          um .md por técnica de Balance/Labeling Strategies (ideia central, algoritmo,
                        assinatura, quando usar, limitações) — ver Documentação/README.md

*/Example_SVHN/        em cada uma dessas pastas, o que é específico do exemplo com o dataset SVHN
                        fica isolado num subdiretório Example_SVHN/ com prefixo Example_, separado
                        do código genérico da biblioteca

Example_SVHN_ActiveLearning.ipynb   notebook de exemplo, na raiz do projeto
```

### Conceitos centrais

**Task** (`Training/tasks.py`) — define como uma tarefa se relaciona com o batch, a loss, a métrica de avaliação e a incerteza por amostra. Implementações prontas:

| Task | Formato do batch | Loss | Incerteza |
|---|---|---|---|
| `ClassificationTask` | `(images, labels)` | `criterion(outputs, labels)` | entropia softmax |
| `SegmentationTask` | `(images, masks)` | `criterion(outputs, masks)` | entropia softmax média por pixel |
| `DetectionTask` | `(images, targets)` — convenção torchvision | `model(images, targets)` devolve o dict de losses | `1 - confiança média` das detecções |
| `VLMContrastiveTask` | `(images, texts)` — pares alinhados | InfoNCE simétrica (estilo CLIP) | margem de similaridade dentro do batch |
| `RegressionTask` | `(inputs, targets)` — alvo contínuo, escalar ou multi-alvo | `criterion(outputs, targets)` (padrão `nn.MSELoss`) | variância entre passadas MC-Dropout (opt-in via `mc_dropout_samples`; sem isso, levanta `NotImplementedError`) |

Cada Task já vem com uma métrica padrão (accuracy, mean IoU, recall@IoU, retrieval top-1, MAE) mas aceita `criterion`/`metric_fn` customizados. Toda Task também implementa `collect_predictions(model, batch, device)` — devolve `(predictions, targets)` já em CPU, usado por `Metrics/task_metrics.py::evaluate_task` para montar relatórios agregados sobre um loader inteiro (ver seção **Metrics** abaixo).

**Query Strategies** — decidem QUAIS amostras do pool não rotulado mandar pro oráculo. Recebem dataloaders (não `dataset` + índices), o que permite plugar qualquer `Dataset`/`transform` próprio. A maioria vive em `Query_Strategies/querys_strategies.py` como funções simples; as três que exigem infraestrutura própria de treino (LPL, VAAL, WAAL) ganharam arquivo dedicado:

- `uncertainty_query_strategy` — usa `task.compute_uncertainty` (por isso recebe uma `task`).
- `margin_query_strategy` — diferença entre as duas classes mais prováveis (menor margem = mais ambíguo).
- `least_confidence_query_strategy` / `variation_ratio_query_strategy` (mesma função, dois nomes) — `1 - max_y p(y|x)`.
- `bald_query_strategy` — incerteza epistêmica via MC-Dropout (T passes estocásticos); exige `nn.Dropout` em algum lugar do modelo.
- `badge_query_strategy` — embeddings de gradiente hipotético (`(p - onehot(ŷ)) ⊗ get_embedding(x)`), batch escolhido via seeding do k-means++ (`sklearn.cluster.kmeans_plusplus`).
- `ClusterMarginQueryStrategy` (classe, não função) — clustering hierárquico (HAC) rodado uma única vez e cacheado, depois margin + amostragem round-robin entre clusters a cada chamada.
- `density_query_strategy` / `diversity_query_strategy` — usam embeddings via `model.get_embedding(x)`, funcionam para qualquer Task desde que o modelo implemente esse método.
- `Query_Strategies/lpl.py` — `LossPredictionModule` + `train_model_with_lpl` (substitui `Training/train.py` nesses ciclos) + `lpl_query_strategy`; exige `model.get_intermediate_features(x)`.
- `Query_Strategies/vaal.py` — `VAE` + `Discriminator` + `train_vaal` + `vaal_query_strategy`; não usa o modelo alvo, só as imagens.
- `Query_Strategies/waal.py` — `WassersteinCritic` (sobre `model.get_embedding`) + `train_model_with_waal` (substitui `Training/train.py`) + `waal_query_strategy`; exige uma `task` com `.criterion`.

LPL/VAAL/WAAL não encaixam no padrão `query_strategy(model, device, budget, unlabeled_loader, unlabeled_indices, ...)` puro porque dependem de um módulo auxiliar treinado à parte (o módulo de loss, o VAE+discriminador, ou o crítico) — amarre-o antes de passar como `query_strategy`, ex.: `query_strategy = lambda **kw: lpl_query_strategy(loss_prediction_module=trained_module, **kw)`.

**Labeling Strategies** — decidem COMO obter o rótulo de uma amostra não rotulada, sem envolver oráculo. Como em Query Strategies, as que funcionam como uma função de seleção ficam em `Labeling_Strategies/labeling_strategies.py`; as que precisam de infraestrutura própria de treino (consistency regularization, teacher-student) ganharam arquivo dedicado — todas assumem classificação de rótulo único, exceto `pseudo_labeling_strategy`/`noisy_student_pseudo_label`, que generalizam pra saídas multi-posição como a do exemplo SVHN.

`Labeling_Strategies/labeling_strategies.py`:

- `pseudo_labeling_strategy` — pseudo-labeling clássico (Lee, 2013): aceita como rótulo a predição do modelo pra toda amostra com confiança acima de um threshold fixo, e delega o balanceamento por classe a um `balance_fn`.
- `flexmatch_strategy` — como acima, mas com um threshold de confiança ADAPTATIVO POR CLASSE (classes que o modelo aprende mais devagar recebem threshold menor).
- `noisy_student_pseudo_label` — threshold bem mais permissivo (o método confia na regularização por ruído do treino seguinte pra compensar); corta/duplica cada classe pra um `max_per_class` fixo.
- `co_training_strategy` / `tri_training_strategy` — usam concordância entre 2 ou 3 modelos (em vez de um threshold de confiança) pra decidir o pseudo-rótulo; Tri-Training também estima e aplica o critério de aceitação por taxa de erro do paper original.
- `PseudoLabeledDataset` — dataset dedicado às amostras pseudo-rotuladas (só lê a imagem do dataset base, nunca o rótulo real); combinado com o conjunto rotulado de verdade via `ConcatDataset` no ciclo, nunca sobrescrevendo um rótulo já existente.

`Labeling_Strategies/augmentation.py` — utilitários compartilhados pelos métodos abaixo: `WeakStrongAugmentDataset`/`MultiAugmentDataset` (datasets que devolvem múltiplas augmentations da mesma imagem), `sharpen`/`mixup` (MixMatch), `ema_update`/`clone_model` (Mean Teacher).

- `Labeling_Strategies/fixmatch.py` — `train_model_with_fixmatch`: pseudo-rótulo hard de uma augmentation fraca, cross-entropy contra a predição da augmentation forte, só acima de um threshold de confiança.
- `Labeling_Strategies/uda.py` — `train_model_with_uda`: como FixMatch, mas com alvo suavizado (sharpening) em vez de hard label, e Training Signal Annealing (TSA) pra não deixar a loss supervisionada convergir rápido demais.
- `Labeling_Strategies/mixmatch.py` — `train_model_with_mixmatch` / `train_model_with_remixmatch`: guessing de rótulo por média+sharpening sobre K augmentations, MixUp entre rotulado e não rotulado; ReMixMatch acrescenta distribution alignment e augmentation anchoring.
- `Labeling_Strategies/mean_teacher.py` — `train_model_with_mean_teacher`: um teacher (EMA dos pesos do student) fornece o alvo de uma consistency loss (MSE) pro student.
- `Labeling_Strategies/vat.py` — `train_model_with_vat`: calcula a perturbação adversarial (via power iteration) que mais muda a predição do modelo, e treina pra ser consistente sob ela — funciona em dados rotulados e não rotulados, sem gerar pseudo-rótulo nenhum.
- `Labeling_Strategies/mpl.py` — `train_model_with_mpl`: teacher e student treinados juntos; o teacher é atualizado via REINFORCE, usando como recompensa o quanto seus pseudo-rótulos melhoraram o student num batch rotulado.

FixMatch/UDA/MixMatch/ReMixMatch/Mean Teacher/VAT/MPL têm loop de treino próprio (substituem `Training/train.py` para os ciclos que os usam) e exigem `task.criterion` — são técnicas de TREINO, não uma chamada única de seleção como as demais.

**Balance Strategies** — como evitar que uma classe domine uma seleção ou um treino. Duas famílias, dois arquivos:

`Balance_Strategies/balance_strategies.py` — operam sobre uma lista de candidatos `(score, index, label)`:

- `class_balance_strategy` — corta cada classe no tamanho da classe com menos candidatos (ou em `max_per_class`, se informado), mantendo os de maior score primeiro.
- `random_oversample_strategy` / `random_undersample_strategy` — duplicam/removem candidatos aleatoriamente até igualar as classes; não precisam de `features`.
- `tomek_links_strategy` / `near_miss_strategy` — undersampling por vizinhança; candidatos precisam vir como 4-tupla `(score, index, label, features)` (ex.: `model.get_embedding(x)`), diferente das duas acima.
- `smote_oversample` / `adasyn_oversample` — geram amostras SINTÉTICAS por interpolação no espaço de embedding (também precisam de `features`); devolvem `(candidates, synthetic)`, onde `synthetic` são pares `(label, embedding)` sem índice real. Para treinar com elas: `SyntheticEmbeddingDataset` + `EmbeddingClassifierWrapper(model)` (exige `model.classify_from_embedding(features)`, já implementado em `SVHNCustomCNN`).

`Balance_Strategies/losses.py` — não filtram candidatos, mudam a `criterion` de uma `Task`:

- `compute_class_weights` / `class_balanced_weights` — pesos por classe (cost-sensitive / effective number of samples) para `nn.CrossEntropyLoss(weight=...)`.
- `FocalLoss` — reduz o peso de exemplos já bem classificados, focando o treino nos difíceis.
- `DynamicCurriculumLoss` — sampling scheduler que decai a distribuição-alvo do batch de desbalanceada pra balanceada ao longo das épocas; exige chamar `.set_epoch(epoch)` a cada época.

**Loop de treino** (`Training/train.py`) — `train_model(model, train_loader, test_loader, task, optimizer, device, epochs)`, agnóstico ao tipo de tarefa; quem sabe como calcular loss/métrica é a `task`.

**Metrics** — duas famílias, dois arquivos, papéis bem diferentes:

`Metrics/task_metrics.py` — avalia a TAREFA em si (não o processo de AL). Diferente de `task.compute_metric` (um valor por amostra de UM batch, leve, usado dentro do loop de treino), aqui as métricas são calculadas sobre a coleção INTEIRA de predições de um loader — necessário pra tudo que não é uma simples média por amostra (precision/recall/F1, matriz de confusão, mAP, R², recall@k...).

- `evaluate_task(model, loader, task, device)` — ponto de entrada único: roda o modelo sobre todo o `loader` (via `task.collect_predictions`) e devolve o relatório certo pro tipo de `task`.
- `classification_report` — accuracy, precision/recall/F1 (macro e weighted, por classe), matriz de confusão. `top_k_accuracy` à parte.
- `regression_report` — MAE, MSE, RMSE, R², MAPE.
- `segmentation_report` — pixel accuracy, mean IoU (global e por classe), mean Dice.
- `detection_report` — precision/recall/F1 por threshold de IoU (correspondência gulosa por score, não a mAP oficial do COCO) + uma aproximação simples de mAP.
- `retrieval_report` — recall@k e MRR (mean reciprocal rank), nas duas direções (imagem→texto e texto→imagem).

`Metrics/al_metrics.py` — avalia o PROCESSO de Active Learning (a query strategy, o custo de cada ciclo), não a tarefa. Já vem plugado em `run_active_learning_cycle`/`run_self_labeling_cycle` (ver abaixo); use direto só se estiver orquestrando um ciclo próprio.

- `ALMetricsTracker` — acumula por ciclo: tempos (`training_time_s`/`selection_time_s`/`classification_time_s`, +`pseudo_labeling_time_s` no self-labeling), `known_classes`, `selected_sample_metric`/`acceptance_rate`, `test_metric`. `.to_csv(path)` grava tudo (sobrescreve, seguro chamar a cada ciclo); `.log_to_tensorboard(writer, cycle)` escreve os campos numéricos como `ALMetrics/<campo>`.
- `selected_sample_metric(model, device, task, loader)` — desempenho do modelo ATUAL (antes de re-treinar) sobre as amostras que a query acabou de escolher, uma vez que o rótulo é conhecido — quantifica quão "informativa" a seleção foi (se o modelo já acertava, a seleção não trouxe muita informação nova).
- `count_known_classes` / `update_known_classes` — quantas classes distintas já apareceram no conjunto rotulado (achatando rótulos multi-posição); a segunda atualiza incrementalmente um `set()`, mais barato que recontar tudo a cada ciclo.

**Ciclo de Active Learning** — duas variantes, para dois cenários diferentes. Em ambas, cada ciclo (re)treina o modelo do zero, avalia em `test_loader`, salva checkpoint + log em TensorBoard/CSV, e registra as métricas de processo acima num `ALMetricsTracker` (parâmetros opcionais `metrics_tracker`/`al_metrics_csv` — se `metrics_tracker` não for informado, um é criado internamente; passe o seu pra inspecionar/plotar os dados depois).

- `run_active_learning_cycle` (`Training/active_learning_cycle.py`) — cenário clássico, com oráculo real: a `query_strategy` roda sobre o pool não rotulado e as amostras escolhidas viram rotuladas com o rótulo verdadeiro do dataset (o oráculo é consultado só para elas). Opcionalmente, com `pseudo_labeling_fn` + `pseudo_dataset_cls` + `pseudo_balance_fn`, também gera pseudo-rótulos sobre o *restante* do pool (o que a query não escolheu) para aumentar o treino do ciclo seguinte.
- `run_self_labeling_cycle` (`Training/self_labeling_cycle.py`) — cenário sem oráculo real: a `query_strategy` escolhe candidatos e o próprio `pseudo_labeling_fn` atua como "oráculo" só para eles (com um `confidence_threshold` mais permissivo, já que não há outra fonte de rótulo). Como não existe rótulo real pras amostras escolhidas, a métrica de processo análoga a `selected_sample_metric` aqui é `acceptance_rate` — fração dos candidatos que o próprio pseudo-labeling aceitou.

#### Diferenças entre `run_active_learning_cycle` e `run_self_labeling_cycle`

| | `run_active_learning_cycle` | `run_self_labeling_cycle` |
|---|---|---|
| Cenário | Existe oráculo real (ex.: humano) | Não existe oráculo real disponível |
| Quem rotula quem a query escolheu | O oráculo — rótulo verdadeiro do dataset | O próprio `pseudo_labeling_fn`, atuando como "oráculo" |
| Escopo do `pseudo_labeling_fn` | Sobre o *restante* do pool, depois da query já ter retirado sua parte | Só sobre os candidatos que a própria query escolheu |
| `confidence_threshold` típico | Alto (padrão `0.95`) — é só um complemento opcional | Mais permissivo (ex. `0.5`) — é a única fonte de rótulo disponível pra quem foi selecionado |
| Persistência do pseudo-rótulo | Transiente: recalculado do zero a cada ciclo, a partir do modelo mais recente | Permanente: gravado uma vez via `on_accepted` (ex.: CSV) e nunca recalculado |
| Como o "rotulado" entra no treino | `Subset(dataset, labeled_indices)` combinado ao pseudo-rotulado via `ConcatDataset`, em memória | `build_labeled_dataset_fn()` reconstrói o dataset do zero a cada ciclo (ex.: relendo o CSV persistido) |
| Candidato que não atinge o `confidence_threshold` | N/A — quem a query escolhe sempre vira rotulado (o oráculo não recusa) | Volta pro pool não rotulado; pode ser escolhido de novo em um ciclo futuro |
| Uso típico | Active Learning clássico, com anotador disponível | Self-training / semi-supervisionado, sem anotador disponível |

### Métodos de referência (roadmap)

Os artigos em `MethodsReferences/` são *surveys* que embasam os métodos que a lib pretende oferecer, um por categoria de estratégia. Cada tabela lista os métodos mais relevantes de cada survey — o que já está implementado e o que é candidato a próxima implementação. Para uma explicação aprofundada de cada método já implementado (ideia central, algoritmo/fórmulas, assinatura, quando usar, limitações), ver os arquivos em [`Documentação/`](Documentação/README.md).

**Query Strategies** — *A Comparative Survey of Deep Active Learning* (Zhan et al., 2022)

| Método | Família | Status |
|---|---|---|
| Entropy / uncertainty softmax | Uncertainty-based | ✅ Implementado (`uncertainty_query_strategy`, via `task.compute_uncertainty`) |
| KMeans | Representative/Diversity-based | ✅ Implementado (`density_query_strategy`) |
| CoreSet / K-Center greedy | Representative/Diversity-based | ✅ Implementado (`diversity_query_strategy`) |
| Margin (Scheffer, Decomain & Wrobel, 2001) | Uncertainty-based | ✅ Implementado (`margin_query_strategy`) |
| Least Confidence (Lewis & Gale, 1994) / Variation Ratio (Freeman, 1965) | Uncertainty-based | ✅ Implementado (`least_confidence_query_strategy` / `variation_ratio_query_strategy`) |
| BALD (Houlsby et al., 2011; MC-Dropout via Gal, Islam & Ghahramani, 2017) | Uncertainty-based | ✅ Implementado (`bald_query_strategy`) |
| BADGE (Ash et al., 2020) | Combinada (uncertainty + diversity) | ✅ Implementado (`badge_query_strategy`) |
| Cluster-Margin (Citovsky et al., 2021) | Representative/Diversity-based | ✅ Implementado (`ClusterMarginQueryStrategy`) |
| Loss Prediction Loss / LPL (Yoo & Kweon, 2019) | Uncertainty-based (model aspect) | ✅ Implementado (`Query_Strategies/lpl.py` — módulo auxiliar + treino conjunto) |
| VAAL (Sinha, Ebrahimi & Darrell, 2019) | Representative/Diversity-based (adversarial) | ✅ Implementado (`Query_Strategies/vaal.py` — VAE + discriminador) |
| WAAL (Shui, Zhou, Gagné & Wang, 2020) | Combinada (adversarial) | ✅ Implementado (`Query_Strategies/waal.py` — crítico Wasserstein + treino conjunto) |

**Labeling Strategies** — *A Review of Pseudo-Labeling for Computer Vision* (Kage, Rothenberger, Andreadis & Diochnos, 2025)

| Método | Família | Status |
|---|---|---|
| Pseudo-Label clássico (Lee, 2013) | Semi-supervisionado, PL original | ✅ Implementado (`pseudo_labeling_strategy`) |
| FlexMatch (Zhang et al., 2021) | Curriculum Pseudo Labeling (threshold adaptativo por classe) | ✅ Implementado (`flexmatch_strategy`) |
| Noisy Student (Xie et al., 2020) | Self-training com ruído | ✅ Implementado (`noisy_student_pseudo_label`) |
| Co-Training (Blum & Mitchell, 1998) | Multi-model (duas visões) | ✅ Implementado (`co_training_strategy`) |
| Tri-Training (Zhou & Li, 2005) | Multi-model (três classificadores) | ✅ Implementado (`tri_training_strategy`) |
| FixMatch (Sohn et al., 2020) | Sample scheduling por métrica (threshold + augmentation fraca/forte) | ✅ Implementado (`Labeling_Strategies/fixmatch.py`) |
| UDA (Xie et al., 2019/2020) | Consistency regularization (+ TSA) | ✅ Implementado (`Labeling_Strategies/uda.py`) |
| MixMatch (Berthelot et al., 2019) | Sample scheduling aleatório + mixing | ✅ Implementado (`Labeling_Strategies/mixmatch.py`) |
| ReMixMatch (Berthelot et al., 2019b) — núcleo (distribution alignment + augmentation anchoring); a loss auxiliar de rotation prediction do paper ficou fora | Sample scheduling aleatório + mixing | ✅ Implementado (`Labeling_Strategies/mixmatch.py`) |
| Mean Teacher (Tarvainen & Valpola, 2017) | Multi-model (EMA teacher-student) | ✅ Implementado (`Labeling_Strategies/mean_teacher.py`) |
| VAT (Miyato et al., 2018) | Consistency regularization (perturbação adversarial) | ✅ Implementado (`Labeling_Strategies/vat.py`) |
| MPL — Meta Pseudo Labels (Pham et al., 2021) | Curriculum/meta-learning (teacher-student bi-level) | ✅ Implementado (`Labeling_Strategies/mpl.py`) |

**Balance Strategies** — *A Comprehensive Survey on Imbalanced Data Learning* (Gao et al., 2025)

| Método | Família | Status |
|---|---|---|
| Balanceamento por classe/quota | Re-labeling (self-training balanceado) | ✅ Implementado (`class_balance_strategy`) |
| Random Oversampling / Undersampling | Data re-balancing | ✅ Implementado (`random_oversample_strategy` / `random_undersample_strategy`) |
| Tomek Links (Tomek, 1976) | Data re-balancing (undersampling por vizinhança) | ✅ Implementado (`tomek_links_strategy`) |
| NearMiss (Mani & Zhang, 2003) | Data re-balancing (undersampling por vizinhança) | ✅ Implementado (`near_miss_strategy`, 3 versões) |
| SMOTE (Chawla et al., 2002) | Data re-balancing (geração sintética) | ✅ Implementado (`smote_oversample`, no espaço de embedding) |
| ADASYN (He et al., 2008) | Data re-balancing (geração sintética adaptativa) | ✅ Implementado (`adasyn_oversample`, no espaço de embedding) |
| Cost-sensitive learning / class weights (Ling & Sheng, 2008) | Feature representation | ✅ Implementado (`compute_class_weights`) |
| Focal Loss (Lin et al., 2017) | Feature representation (nível de instância) | ✅ Implementado (`FocalLoss`) |
| Class-Balanced Loss (Cui et al., 2019 — effective number of samples) | Feature representation | ✅ Implementado (`class_balanced_weights`) |
| Dynamic Curriculum Learning (Wang et al., 2019) — só o núcleo (sampling scheduler + DSL loss); a loss auxiliar de metric learning (triplet+easy anchors) do paper ficou fora, por ser uma técnica de representação ortogonal a balanceamento | Feature representation | ✅ Implementado (`DynamicCurriculumLoss`) |

### Principais tecnologias

- Python
- PyTorch / torchvision
- scikit-learn (KMeans, usado pela estratégia de densidade)
- Dataset de exemplo: Street View House Numbers (SVHN)
