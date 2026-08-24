# MPL — Meta Pseudo Labels (`train_model_with_mpl`)

**Categoria:** Curriculum/meta-learning (teacher-student bi-level)
**Arquivo:** `Labeling_Strategies/mpl.py`
**Referência:** Pham, H., Dai, Z., Xie, Q., Luong, M.-T. & Le, Q. V. (2021). *"Meta Pseudo Labels"*. CVPR.

## Ideia central

Diferente de Pseudo-Label clássico e Noisy Student (onde o teacher é **fixo** durante o treino do student), em MPL o teacher é treinado **em paralelo** com o student, usando como sinal de recompensa o quanto seus pseudo-rótulos ajudaram o student a melhorar num batch rotulado. É uma otimização em dois níveis (bi-level): o student otimiza a loss não supervisionada (usando pseudo-rótulos do teacher); o teacher otimiza a loss supervisionada **do student**, indiretamente, através de como sua escolha de pseudo-rótulo afetou a atualização do student. Requer **dois** modelos com a mesma arquitetura mas pesos independentes.

## Algoritmo (um passo, `train_step_mpl`)

Versão prática com rótulos "hard" (Seção 2 do paper, "we sample the hard pseudo labels... rely on a modified version of REINFORCE"):

1. O teacher prediz no batch não rotulado e **amostra** um pseudo-rótulo hard de sua distribuição (`torch.multinomial` — amostragem categórica, não `argmax`; necessário para o REINFORCE, que precisa de uma ação estocástica com probabilidade associada).
2. O student dá **um** passo de gradiente real usando esses pseudo-rótulos:
   ```
   θ'_S = θ_S - η_S·∇_θ_S L_u(θ_T,θ_S),   L_u = CE(pseudo_labels, S(x_u;θ_S))
   ```
3. Mede a loss supervisionada do student (**já atualizado**) num batch **rotulado** — a diferença em relação à loss **antes** do passo é a "recompensa": negativa (loss caiu) significa que o pseudo-rótulo ajudou.
4. O teacher é atualizado via **REINFORCE aproximado**, reforçando (ou penalizando) a probabilidade que ele deu ao pseudo-rótulo amostrado, proporcionalmente à recompensa:
   ```
   reward = L_l(θ_S) - L_l(θ'_S)
   teacher_loss = -reward · log T(ŷ_u | x_u; θ_T)          (política REINFORCE)
   ```

## Assinaturas

```python
train_step_mpl(teacher, student, teacher_optimizer, student_optimizer,
               labeled_batch, unlabeled_batch, task, device, temperature=1.0) -> dict de losses/reward

train_model_with_mpl(teacher, student, labeled_loader, unlabeled_loader, test_loader, task,
                      teacher_optimizer, student_optimizer, device, epochs, temperature=1.0,
                      writer=None, log_prefix="") -> (loss_u, test_metric)

finetune_student_on_labeled(student, labeled_loader, task, optimizer, device, steps) -> student
```

- Requer `task` com atributo `.criterion`.
- `train_model_with_mpl` chama `train_step_mpl` a cada par de batches (rotulado, não rotulado), reciclando o loader menor; a **avaliação** roda sobre o **student** (é ele quem se quer usar no final — o teacher existe só para gerar pseudo-rótulos melhores).
- `temperature`: suaviza a distribuição do teacher antes da amostragem categórica (`softmax(logits/temperature)`).

## `finetune_student_on_labeled` — passo final opcional

"*Since the student in Meta Pseudo Labels only learns from unlabeled data with pseudo labels... we can take a student model that has converged after training with Meta Pseudo Labels and finetune it on labeled data*" — trecho do paper que motiva esta função auxiliar: um fine-tuning curto e direto (supervisionado padrão, sem nada de meta-learning) só nos dados rotulados de verdade, para corrigir qualquer viés herdado dos pseudo-rótulos ao longo do treino principal.

## Quando usar

Quando há orçamento computacional para treinar dois modelos simultaneamente com dois otimizadores independentes, e o cenário se beneficia de um teacher que se adapta ao progresso do student (diferente de Noisy Student, onde o teacher é congelado e todo o processo iterativo acontece por fora, em rodadas completas de re-treino). É a técnica de treino mais complexa em termos de loop da biblioteca — mistura RL (REINFORCE) com supervisão padrão no mesmo passo.

## Limitações

- REINFORCE é um estimador de gradiente de alta variância — o sinal de treino do teacher pode ser ruidoso, especialmente em batches pequenos.
- Custo computacional por passo é maior que as demais técnicas: cada passo faz duas passadas pelo student (antes e depois da atualização) e duas pelo teacher (predição inicial + gradiente da política), além de dois otimizadores para manter sincronizados.
- A recompensa é calculada sobre um único batch rotulado por passo — sensível a quão representativo esse batch é do desempenho real do student.
