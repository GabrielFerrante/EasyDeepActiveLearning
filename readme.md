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
Query_Strategies/     estratégias de seleção de amostras pro oráculo (Query_Strategies/querys_strategies.py)
Labeling_Strategies/  estratégias de rotulação, ex. pseudo-labeling (Labeling_Strategies/labeling_strategies.py)
Balance_Strategies/   estratégias de balanceamento por classe (Balance_Strategies/balance_strategies.py)
Training/              loop de treino, ciclo de Active Learning e as Tasks (Training/tasks.py)
Utils/                 utilitários gerais
MethodsReferences/    artigos de referência (surveys) que embasam os métodos da lib

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

Cada Task já vem com uma métrica padrão (accuracy, mean IoU, recall@IoU, retrieval top-1) mas aceita `criterion`/`metric_fn` customizados.

**Query Strategies** (`Query_Strategies/querys_strategies.py`) — decidem QUAIS amostras do pool não rotulado mandar pro oráculo. Recebem dataloaders (não `dataset` + índices), o que permite plugar qualquer `Dataset`/`transform` próprio:

- `uncertainty_query_strategy` — usa `task.compute_uncertainty` (por isso recebe uma `task`).
- `density_query_strategy` / `diversity_query_strategy` — usam embeddings via `model.get_embedding(x)`, funcionam para qualquer Task desde que o modelo implemente esse método.

**Labeling Strategies** (`Labeling_Strategies/labeling_strategies.py`) — decidem COMO obter o rótulo de uma amostra não rotulada, sem envolver oráculo:

- `pseudo_labeling_strategy` — pseudo-labeling clássico (Lee, 2013): aceita como rótulo a predição do modelo pra toda amostra com confiança acima de um threshold, e delega o balanceamento por classe a um `balance_fn`.
- `PseudoLabeledDataset` — dataset dedicado às amostras pseudo-rotuladas (só lê a imagem do dataset base, nunca o rótulo real); combinado com o conjunto rotulado de verdade via `ConcatDataset` no ciclo, nunca sobrescrevendo um rótulo já existente.

**Balance Strategies** (`Balance_Strategies/balance_strategies.py`) — dado um conjunto de candidatos `(score, index, label)`, decidem como evitar que uma classe domine a seleção:

- `class_balance_strategy` — corta cada classe no tamanho da classe com menos candidatos (ou em `max_per_class`, se informado), mantendo os de maior score primeiro.

**Loop de treino** (`Training/train.py`) — `train_model(model, train_loader, test_loader, task, optimizer, device, epochs)`, agnóstico ao tipo de tarefa; quem sabe como calcular loss/métrica é a `task`.

**Ciclo de Active Learning** — duas variantes, para dois cenários diferentes. Em ambas, cada ciclo (re)treina o modelo do zero, avalia em `test_loader` e salva checkpoint + log em TensorBoard/CSV.

- `run_active_learning_cycle` (`Training/active_learning_cycle.py`) — cenário clássico, com oráculo real: a `query_strategy` roda sobre o pool não rotulado e as amostras escolhidas viram rotuladas com o rótulo verdadeiro do dataset (o oráculo é consultado só para elas). Opcionalmente, com `pseudo_labeling_fn` + `pseudo_dataset_cls` + `pseudo_balance_fn`, também gera pseudo-rótulos sobre o *restante* do pool (o que a query não escolheu) para aumentar o treino do ciclo seguinte.
- `run_self_labeling_cycle` (`Training/self_labeling_cycle.py`) — cenário sem oráculo real: a `query_strategy` escolhe candidatos e o próprio `pseudo_labeling_fn` atua como "oráculo" só para eles (com um `confidence_threshold` mais permissivo, já que não há outra fonte de rótulo).

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

Os artigos em `MethodsReferences/` são *surveys* que embasam os métodos que a lib pretende oferecer, um por categoria de estratégia. Cada tabela lista os métodos mais relevantes de cada survey — o que já está implementado e o que é candidato a próxima implementação.

**Query Strategies** — *A Comparative Survey of Deep Active Learning* (Zhan et al., 2022)

| Método | Família | Status |
|---|---|---|
| Entropy / uncertainty softmax | Uncertainty-based | ✅ Implementado (`uncertainty_query_strategy`, via `task.compute_uncertainty`) |
| KMeans | Representative/Diversity-based | ✅ Implementado (`density_query_strategy`) |
| CoreSet / K-Center greedy | Representative/Diversity-based | ✅ Implementado (`diversity_query_strategy`) |
| Margin | Uncertainty-based | ⬜ Planejado |
| Least Confidence / VarRatio | Uncertainty-based | ⬜ Planejado |
| BALD (Bayesian AL by Disagreement) | Uncertainty-based | ⬜ Planejado |
| Loss Prediction Loss (LPL) | Uncertainty-based (model aspect) | ⬜ Planejado |
| BADGE (gradient embeddings) | Combinada (uncertainty + diversity) | ⬜ Planejado |
| Cluster-Margin | Representative/Diversity-based | ⬜ Planejado |
| VAAL / WAAL (adversarial AL) | Representative/Diversity-based | ⬜ Planejado |

**Labeling Strategies** — *A Review of Pseudo-Labeling for Computer Vision* (Kage, Rothenberger, Andreadis & Diochnos, 2025)

| Método | Família | Status |
|---|---|---|
| Pseudo-Label clássico (Lee, 2013) | Semi-supervisionado, PL original | ✅ Implementado (`pseudo_labeling_strategy`) |
| FixMatch | Sample scheduling por métrica (threshold + augmentation fraca/forte) | ⬜ Planejado |
| MixMatch / ReMixMatch | Sample scheduling aleatório + mixing | ⬜ Planejado |
| Mean Teacher / Noisy Student | Multi-model (teacher-student) | ⬜ Planejado |
| Co-Training / Tri-Training | Multi-model (múltiplos modelos concordando) | ⬜ Planejado |
| UDA / VAT | Consistency regularization | ⬜ Planejado |
| FlexMatch / MPL (Meta Pseudo Labels) | Curriculum/meta-learning | ⬜ Planejado |

**Balance Strategies** — *A Comprehensive Survey on Imbalanced Data Learning* (Gao et al., 2025)

| Método | Família | Status |
|---|---|---|
| Balanceamento por classe/quota | Re-labeling (self-training balanceado) | ✅ Implementado (`class_balance_strategy`) |
| Random over/undersampling | Data re-balancing | ⬜ Planejado |
| SMOTE / ADASYN | Data re-balancing (geração sintética) | ⬜ Planejado |
| Tomek Links / NearMiss | Data re-balancing (undersampling por vizinhança) | ⬜ Planejado |
| Cost-sensitive learning / class weights | Feature representation | ⬜ Planejado |
| Focal Loss | Feature representation (nível de instância) | ⬜ Planejado |
| Class-Balanced Loss (effective number of samples) | Feature representation | ⬜ Planejado |

### Principais tecnologias

- Python
- PyTorch / torchvision
- scikit-learn (KMeans, usado pela estratégia de densidade)
- Dataset de exemplo: Street View House Numbers (SVHN)
