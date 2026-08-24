# Cost-Sensitive Learning / Class Weights (`compute_class_weights`)

**Categoria:** Feature representation (loss-reweighting)
**Arquivo:** `Balance_Strategies/losses.py`
**Referência:** Ling, C. X. & Sheng, V. S. (2008). *"Cost-Sensitive Learning and the Class Imbalance Problem"*, Encyclopedia of Machine Learning; He, H. & Garcia, E. A. (2009). *"Learning from Imbalanced Data"*, IEEE TKDE.

## Ideia central

Diferente de todas as estratégias anteriores (que decidem **quais/quantas** amostras entram na seleção ou no treino), cost-sensitive learning não mexe nos dados — muda **como a função de perda pondera cada classe**. Classes minoritárias recebem um peso maior na loss, então cada erro nelas custa proporcionalmente mais ao otimizador do que um erro na classe majoritária, compensando o desbalanceamento sem duplicar nem descartar nenhuma amostra.

## Fórmulas

A biblioteca implementa dois esquemas:

- **`inverse_frequency`**: `w_c = N / (num_classes · n_c)`, onde `N` é o total de amostras e `n_c` é a contagem da classe `c`. Quanto menos amostras a classe tem, maior o peso; a média dos pesos, ponderada pela frequência real de cada classe, é 1 (não distorce a escala geral da loss).
- **`inverse_sqrt`**: `w_c = 1/√n_c`, renormalizado para média 1. Correção mais suave que `inverse_frequency` — útil quando o esquema padrão super-corrige em cenários de desbalanceamento muito extremo (uma classe com poucas dezenas de amostras contra uma com dezenas de milhares).

## Assinatura

```python
compute_class_weights(labels, num_classes, scheme="inverse_frequency") -> torch.Tensor
```

- `labels`: array/tensor 1D com o rótulo de cada amostra do conjunto de treino — para saídas multi-posição (ex.: SVHN, `[Batch, N_posições]`), é preciso achatar antes de chamar (a própria função já faz `np.asarray(labels).flatten()` internamente).
- `scheme`: `"inverse_frequency"` (padrão) ou `"inverse_sqrt"`.
- Retorno: tensor de pesos por classe, pronto para usar em `nn.CrossEntropyLoss(weight=pesos)` ou como `alpha` de `FocalLoss`.

## Como usar na biblioteca

```python
weights = compute_class_weights(all_train_labels, num_classes=11)
task = ClassificationTask(criterion=nn.CrossEntropyLoss(weight=weights))
```

`Balance_Strategies/losses.py` não filtra candidatos — os pesos entram diretamente na `criterion` de uma `Task` (ver `Training/tasks.py`).

## Quando usar

É a técnica mais simples e barata de correção de desbalanceamento no nível da loss: um único cálculo antes do treino, sem custo por batch. Boa primeira tentativa antes de partir para Focal Loss ou Class-Balanced Loss, que fazem suposições mais específicas sobre a natureza do desbalanceamento (respectivamente: exemplos difíceis vs. fáceis; redundância entre amostras da mesma classe).

## Limitações

- Pondera toda a classe uniformemente — não diferencia amostras fáceis de difíceis dentro da mesma classe (ver Focal Loss para essa distinção).
- `inverse_frequency` pode super-corrigir em desbalanceamentos extremos, fazendo o gradiente da classe minoritária dominar demais o treino (daí a opção `inverse_sqrt`, mais suave).
