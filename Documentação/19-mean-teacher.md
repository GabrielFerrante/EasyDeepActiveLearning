# Mean Teacher (`train_model_with_mean_teacher`)

**Categoria:** Multi-model (EMA teacher-student)
**Arquivo:** `Labeling_Strategies/mean_teacher.py`
**Referência:** Tarvainen, A. & Valpola, H. (2017). *"Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results"*. NeurIPS.

## Ideia central

Mantém dois modelos: um **student**, treinado normalmente por gradiente, e um **teacher**, cujos pesos são a **média móvel exponencial (EMA)** dos pesos do student ao longo do treino — o teacher nunca recebe gradiente diretamente, só a atualização EMA depois de cada passo do student. A ideia é que a média de vários pontos do trajeto de otimização do student tende a ser um modelo **mais estável e melhor calibrado** do que qualquer snapshot individual — o teacher fornece um alvo de consistência mais confiável do que o próprio student em qualquer passo específico.

## Fórmulas

```
θ'_t = ema_decay · θ'_{t-1} + (1 - ema_decay) · θ_t        — EMA do teacher
J(θ) = E[ ||f(x,θ',η') - f(x,θ,η)||² ]                      — consistency loss (MSE)
loss = classification_cost(student) + consistency_weight · J(θ)
```

O "ruído" `η`/`η'` do paper (augmentation/dropout independentes para student e teacher) vem, nesta implementação, do próprio dropout de cada modelo em `model.train()` — mesma imagem, duas passadas (student com gradiente, teacher sem) — uma simplificação comum em implementações públicas do método, que dispensa augmentation dupla explícita.

A consistency loss é aplicada **tanto no rotulado quanto no não rotulado** (Fig. 2 do paper); a classification loss só no rotulado.

## Assinatura

```python
train_model_with_mean_teacher(student, teacher, labeled_loader, unlabeled_loader, test_loader, task,
                               optimizer, device, epochs, ema_decay=0.999, consistency_weight=1.0,
                               consistency_rampup_steps=None, writer=None, log_prefix="") -> (loss, test_metric)
```

- `teacher`: uma cópia do `student` (ver `Labeling_Strategies.augmentation.clone_model`) — nunca é otimizado diretamente, só via `ema_update` (`augmentation.py`) depois de cada `optimizer.step()`.
- `ema_decay`: fator de EMA (padrão `0.999`, valor típico da literatura — quanto mais perto de 1, mais devagar o teacher "esquece" pesos antigos).
- `consistency_rampup_steps`: se informado, o peso da consistency loss sobe linearmente de 0 até `consistency_weight` ao longo desses passos — o paper usa um ramp-up (sigmoide) nos primeiros passos, já que no início o teacher (recém-inicializado, igual ao student) ainda não tem alvos confiáveis diferentes do próprio student.
- Requer `task` com atributo `.criterion`.
- A avaliação final (`_evaluate`) roda sobre o **teacher**, não o student — é ele quem tipicamente generaliza melhor, sendo o "produto final" do método.

## Quando usar

Quando é viável manter dois modelos em memória simultaneamente (custo de memória ~2x um único modelo) e o objetivo é estabilidade de treino — o teacher, por ser uma média temporal, tende a ter previsões menos ruidosas que qualquer snapshot do student, o que se traduz em um alvo de consistência mais confiável do que, por exemplo, comparar o modelo consigo mesmo em duas passadas (auto-ensembling sem EMA).

## Limitações

- Custo de memória ~2x (dois modelos completos), mais o custo de duas passadas forward por passo (student e teacher).
- A "fonte de ruído" simplificada (dropout implícito em vez de augmentation dupla explícita) é uma aproximação de implementações públicas, não uma leitura literal do paper — pode reduzir o efeito de regularização em modelos com pouco ou nenhum dropout na arquitetura.
- `ema_decay` alto demais faz o teacher reagir devagar a melhorias reais do student; baixo demais aproxima o comportamento de um teacher sem EMA (perde o benefício de estabilidade).
