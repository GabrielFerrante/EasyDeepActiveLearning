# Dynamic Curriculum Learning (`DynamicCurriculumLoss`)

**Categoria:** Feature representation
**Arquivo:** `Balance_Strategies/losses.py`
**Referência:** Wang, Y., Gan, W., Yang, J., Wu, W. & Yan, J. (2019). *"Dynamic Curriculum Learning for Imbalanced Data Classification"*. ICCV.

## Ideia central

O paper propõe treinar em duas fases implícitas, controladas por um **scheduler**: no início do treino, o modelo vê a distribuição de classes **real** (desbalanceada) — é mais fácil aprender representações gerais assim, com mais dados das classes majoritárias. Conforme o treino avança, a distribuição efetiva vista pelo modelo é empurrada gradualmente para **balanceada**, para refinar a fronteira de decisão sem que a classe majoritária domine as últimas épocas. A transição de "desbalanceado" para "balanceado" é controlada por uma função de scheduler `g(l)` que decai de 1 para 0 ao longo das épocas.

O paper tem dois componentes: (1) o **Sampling Scheduler + Dynamic Selective Learning (DSL) loss** — a parte diretamente ligada a balanceamento, implementada aqui — e (2) uma loss auxiliar de metric learning (*Triplet Loss with Easy Anchors*), voltada para melhorar a qualidade do embedding aprendido. **O item (2) fica fora do escopo desta implementação**, por ser uma técnica de aprendizado de representação (metric learning) ortogonal a balanceamento propriamente dito — só o núcleo (1) foi implementado.

## Fórmulas

Distribuição-alvo na época `l` (de um total de `total_epochs`):

```
D_target(l) = D_train ^ g(l)                    (Eq. 6)
```

onde `D_train` é a distribuição real de classes do dataset de treino inteiro (normalizada pela classe minoritária, para deixar os valores manejáveis), e `g(l)` é uma das quatro funções de scheduler:

- `scheduler_cos` (Eq. 1): `cos((l/total_epochs)·π/2)` — decai rápido no início, devagar no fim.
- `scheduler_linear` (Eq. 2): `1 - l/total_epochs` — decaimento constante.
- `scheduler_exp` (Eq. 3, com `lam=0.9` por padrão): `lam^l` — decai devagar no início, rápido no fim.
- `scheduler_composite` (Eq. 4, padrão da classe, exemplo principal do paper): `0.5·cos((l/total_epochs)·π) + 0.5` — rápido, depois devagar, depois rápido de novo.

Comparando `D_target(l)` com a distribuição real do **batch atual** (`D_current`), cada classe `j` recebe um peso por amostra (Eq. 8):

- Se `D_target,j(l) / D_current,j >= 1` (classe rara *nesse batch específico*): todas as amostras dessa classe no batch recebem peso igual a essa razão (upweight).
- Se `< 1` (classe super-representada *nesse batch*): cada amostra da classe é mantida com peso 1 com probabilidade igual a essa razão, e descartada (peso 0) caso contrário — uma reamostragem estocástica por batch.

A loss final (Eq. 7) é normalizada pelo **tamanho fixo do batch `N`** (não pela contagem de amostras que sobraram após a reamostragem estocástica):

```
L_DSL = -(1/N) Σ_j Σ_i w_j · log p(...)
```

`D_current` (a distribuição real do batch atual) usa como denominador de normalização a contagem da classe **minoritária presente no batch** — uma classe totalmente ausente do batch não deve entrar nesse mínimo (ela não tem contagem "pequena", ela tem contagem zero, o que é um caso diferente).

## Assinatura

```python
DynamicCurriculumLoss(class_counts, total_epochs, scheduler_fn=scheduler_composite, base_criterion=None)
loss_fn.set_epoch(epoch)          # OBRIGATÓRIO a cada época
loss = loss_fn(outputs, targets)  # outputs: [Batch, ..., C]   targets: [Batch, ...]
```

- `class_counts`: contagem de amostras por classe no dataset de treino **inteiro** (define `D_train`).
- `total_epochs`: total de épocas do treino (usado para normalizar `l/total_epochs` no scheduler).
- `scheduler_fn`: uma de `scheduler_cos`/`scheduler_linear`/`scheduler_exp`/`scheduler_composite`.
- `base_criterion`: loss por amostra a ponderar (padrão: `nn.CrossEntropyLoss(reduction="none")`); precisa aceitar `reduction="none"` e devolver um valor por amostra.

**Requer chamar `.set_epoch(epoch)` a cada época de treino** — `Training/train.py:train_model` não sabe disso sozinho; quem orquestra o loop precisa chamar (por exemplo, um wrapper em volta de `train_model`, ou uma modificação direta no loop de treino).

## Quando usar

Quando o treino é longo o suficiente (várias dezenas de épocas) para que a transição gradual do scheduler faça diferença — em treinos curtos, o efeito prático se aproxima de aplicar Class-Balanced Loss ou class weights fixos desde o início, já que a distribuição-alvo não tem tempo de se afastar muito da real.

## Limitações

- Só o núcleo de balanceamento foi implementado; a loss auxiliar de metric learning do paper (que ajuda a formar clusters de embedding mais compactos por classe) não está incluída.
- Exige orquestração externa (`.set_epoch`) — não é plug-and-play como as demais losses da biblioteca, que só precisam ser passadas como `criterion` de uma `Task`.
- A escolha do `scheduler_fn` é um hiperparâmetro adicional sem uma regra fixa de qual usar — o paper reporta `scheduler_composite` como o exemplo principal, mas não como universalmente melhor.

## Nota de correção

Uma versão anterior desta implementação tinha dois bugs, já corrigidos: (1) a loss era normalizada pela contagem de amostras não-descartadas em vez do tamanho fixo `N` do batch (Eq. 7), o que inflava a escala da loss quanto mais amostras a reamostragem estocástica descartava num batch; (2) o cálculo do mínimo de `D_current` aplicava um `clamp(min=1)` sobre a contagem de TODAS as classes antes de tirar o mínimo — uma classe ausente do batch (contagem 0) virava artificialmente 1 e quase sempre acabava sendo o "mínimo", colapsando a normalização de `D_current` e distorcendo a razão da Eq. 8 para todas as classes presentes. A correção calcula o mínimo só entre as classes efetivamente presentes no batch atual.
