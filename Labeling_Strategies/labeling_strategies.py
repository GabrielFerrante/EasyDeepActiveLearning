import torch
import numpy as np
from torch.utils.data import Dataset

# Diferente das Query_Strategies (que escolhem QUAIS amostras mandar pro oráculo), as estratégias
# daqui decidem COMO obter o rótulo de uma amostra não rotulada. A pseudo-labeling clássica não
# envolve oráculo nenhum: usa a própria predição do modelo como rótulo, quando ele está confiante o
# suficiente.


def pseudo_labeling_strategy(model, device, unlabeled_loader, unlabeled_indices, balance_fn,
                              confidence_threshold=0.95, max_per_class=None, **kwargs):
    """
    Pseudo-labeling clássico (Lee, 2013): roda o modelo sobre o pool não rotulado e junta como
    candidato a pseudo-rótulo a predição (argmax do softmax) de toda amostra cuja confiança
    (probabilidade máxima) seja >= confidence_threshold. O balanceamento por classe entre os
    candidatos aprovados é feito por balance_fn (ver Balance_Strategies/balance_strategies.py) — sem
    isso, a classe que o modelo já acerta mais (logo, mais confiante) tende a dominar o conjunto
    pseudo-rotulado e reforçar esse viés a cada ciclo.

    balance_fn: callable(candidates, max_per_class) -> candidatos balanceados, onde candidates é uma
    lista de (confidence, dataset_index, pred). Ex.: Balance_Strategies.balance_strategies.class_balance_strategy.

    unlabeled_loader deve ser um DataLoader com shuffle=False, iterando na mesma ordem de
    unlabeled_indices (ver Query_Strategies/querys_strategies.py).

    Devolve (selected_indices, pseudo_labels): os índices originais do dataset aceitos e os rótulos
    previstos correspondentes (na mesma ordem), já balanceados por classe.
    """
    model.eval()
    candidates = []  # [(confidence, dataset_index, pred_tensor), ...]

    position = 0
    with torch.no_grad():
        for batch in unlabeled_loader:
            images = batch[0].to(device)
            outputs = model(images)  # [Batch, ..., num_classes]

            probs = torch.softmax(outputs, dim=-1)
            per_position_confidence, preds = torch.max(probs, dim=-1)

            # Reduz eventuais dimensões extras (ex.: várias posições/heads) para uma confiança por amostra
            sample_confidence = per_position_confidence
            while sample_confidence.dim() > 1:
                sample_confidence = torch.mean(sample_confidence, dim=-1)

            batch_size = images.size(0)
            for i in range(batch_size):
                confidence = sample_confidence[i].item()
                if confidence >= confidence_threshold:
                    candidates.append((confidence, unlabeled_indices[position + i], preds[i].cpu()))
            position += batch_size

    balanced = balance_fn(candidates, max_per_class=max_per_class)

    selected_indices = [index for _, index, _ in balanced]
    pseudo_labels = [pred for _, _, pred in balanced]
    return np.array(selected_indices, dtype=int), pseudo_labels


def flexmatch_strategy(model, device, unlabeled_loader, unlabeled_indices, balance_fn,
                        num_classes, tau=0.95, use_warmup=True, max_per_class=None, **kwargs):
    """
    FlexMatch — Curriculum Pseudo Labeling (Zhang, Wang, Hou, Wu, Wang, Okumura & Shinozaki, 2021,
    NeurIPS, "FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling"):
    extensão do pseudo-labeling clássico com um threshold de confiança ADAPTATIVO POR CLASSE, em vez
    de um único threshold fixo pra todas.

    Ideia central: classes "difíceis" (poucas amostras não rotuladas atingem tau com confiança) têm
    seu threshold REDUZIDO, deixando mais amostras dessa classe entrarem; classes "fáceis" mantêm o
    threshold próximo de tau. Evita que o modelo ignore sistematicamente as classes que aprende mais
    devagar.

        σ(c) = |{u : argmax p(y|u) = c  e  max p(y|u) > tau}|              (Eq. 5)
        β(c) = σ(c) / max_c σ(c)                                          (Eq. 6, sem warm-up)
        T(c) = β(c) · tau                                                 (Eq. 7)

    Com use_warmup=True (recomendado; Eq. 11), o denominador de β(c) também conta as amostras ainda
    "não usadas" por nenhuma classe (max(max_c σ, N - Σ_c σ)) — evita confiar demais na estimativa
    de aprendizado nas primeiras iterações, quando as poucas amostras confiantes podem estar
    concentradas por acaso numa única classe.

    Assume classificação de rótulo único (saída [Batch, num_classes]) — o conceito de "threshold por
    classe" não generaliza diretamente pra saídas multi-posição como a do exemplo SVHN.

    Recalculado do zero a cada chamada, a partir das predições atuais do modelo sobre TODO o
    unlabeled_loader (equivalente a uma iteração `t` do Algorithm 1 do paper).

    Devolve (selected_indices, pseudo_labels), igual a pseudo_labeling_strategy.
    """
    model.eval()
    confidences, preds = [], []
    with torch.no_grad():
        for batch in unlabeled_loader:
            images = batch[0].to(device)
            probs = torch.softmax(model(images), dim=-1)  # [Batch, num_classes]
            batch_confidence, batch_preds = torch.max(probs, dim=-1)
            confidences.append(batch_confidence.cpu())
            preds.append(batch_preds.cpu())

    confidences = torch.cat(confidences)
    preds = torch.cat(preds)

    above_tau = confidences > tau
    sigma = torch.zeros(num_classes)
    for c in range(num_classes):
        sigma[c] = ((preds == c) & above_tau).sum().float()

    if use_warmup:
        denom = max(sigma.max().item(), len(preds) - sigma.sum().item())
    else:
        denom = sigma.max().item()
    denom = max(denom, 1.0)  # evita divisão por zero antes de qualquer amostra passar de tau

    class_thresholds = (sigma / denom) * tau  # T(c), Eq. 7

    candidates = []
    for i in range(len(preds)):
        c = preds[i].item()
        if confidences[i].item() > class_thresholds[c].item():
            candidates.append((confidences[i].item(), unlabeled_indices[i], preds[i]))

    balanced = balance_fn(candidates, max_per_class=max_per_class)
    selected_indices = [index for _, index, _ in balanced]
    pseudo_labels = [pred for _, _, pred in balanced]
    return np.array(selected_indices, dtype=int), pseudo_labels


def noisy_student_pseudo_label(model, device, unlabeled_loader, unlabeled_indices,
                                confidence_threshold=0.3, max_per_class=None, soft=False, **kwargs):
    """
    Noisy Student — passo de geração de pseudo-rótulos (Xie, Luong, Hovy & Le, 2020, CVPR, "Self-
    training with Noisy Student improves ImageNet classification"): o TEACHER (modelo já treinado só
    com dados rotulados, ou de uma iteração anterior) prediz sobre o pool não rotulado; mantém
    amostras com confiança acima de confidence_threshold (o paper usa 0.3 — bem mais permissivo que
    o 0.95 do pseudo-labeling clássico, já que o método confia na regularização por ruído do STUDENT
    pra compensar rótulos ruidosos) e, se max_per_class for informado, corta/duplica cada classe pra
    esse tamanho (o paper limita a 130K por classe e duplica aleatoriamente classes com poucas
    imagens até atingir a cota, pra manter o treino do student balanceado).

    Esse é só o passo de SELEÇÃO — a parte "Noisy" do método está em COMO o student é treinado
    depois, não em como os pseudo-rótulos são escolhidos: treine o student (igual ou MAIOR que o
    teacher) sobre rotulado + pseudo-rotulado com ruído injetado tanto na entrada (augmentation
    forte, ex. RandAugment — ver Labeling_Strategies.augmentation) quanto no modelo (dropout,
    stochastic depth — configure seu model_fn com isso se quiser replicar o ruído de modelo também).
    Repita o processo (o student vira o teacher da próxima iteração) pra fidelidade completa ao paper.

    soft: se True, devolve a distribuição de probabilidade inteira como "pseudo-rótulo" em vez do
    argmax (o paper reporta desempenho parecido entre hard/soft, com soft levemente melhor pra dados
    fora do domínio de treino).

    Devolve (selected_indices, pseudo_labels), igual a pseudo_labeling_strategy.
    """
    model.eval()
    candidates_by_class = {}

    position = 0
    with torch.no_grad():
        for batch in unlabeled_loader:
            images = batch[0].to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=-1)
            confidence, preds = torch.max(probs, dim=-1)

            while confidence.dim() > 1:
                confidence = torch.mean(confidence, dim=-1)

            batch_size = images.size(0)
            for i in range(batch_size):
                if confidence[i].item() >= confidence_threshold:
                    label = probs[i].cpu() if soft else preds[i].cpu()
                    key = tuple(preds[i].cpu().tolist()) if preds[i].dim() > 0 else preds[i].item()
                    candidates_by_class.setdefault(key, []).append(
                        (confidence[i].item(), unlabeled_indices[position + i], label)
                    )
            position += batch_size

    if not candidates_by_class:
        return np.array([], dtype=int), []

    rng = np.random.default_rng(42)
    target_count = max_per_class or max(len(g) for g in candidates_by_class.values())

    selected_indices, pseudo_labels = [], []
    for group in candidates_by_class.values():
        chosen = list(group)
        if len(chosen) > target_count:
            chosen.sort(key=lambda c: c[0], reverse=True)
            chosen = chosen[:target_count]
        elif len(chosen) < target_count:
            extra_idx = rng.integers(0, len(chosen), size=target_count - len(chosen))
            chosen = chosen + [group[i] for i in extra_idx]  # duplica pra balancear, como no paper

        for _, index, label in chosen:
            selected_indices.append(index)
            pseudo_labels.append(label)

    return np.array(selected_indices, dtype=int), pseudo_labels


def _confident_predictions(model, device, loader):
    model.eval()
    confidences, preds = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device)
            probs = torch.softmax(model(images), dim=-1)
            batch_confidence, batch_preds = torch.max(probs, dim=-1)
            while batch_confidence.dim() > 1:
                batch_confidence = torch.mean(batch_confidence, dim=-1)
            confidences.append(batch_confidence.cpu())
            preds.append(batch_preds.cpu())
    return torch.cat(confidences), torch.cat(preds)


def co_training_strategy(model_a, model_b, device, unlabeled_loader_a, unlabeled_loader_b,
                          unlabeled_indices, confidence_threshold=0.95, **kwargs):
    """
    Co-Training (Blum & Mitchell, 1998, COLT, "Combining Labeled and Unlabeled Data with Co-
    Training"): duas visões independentes dos mesmos dados treinam dois classificadores; cada um
    rotula, para as amostras em que está confiante, exemplos pro OUTRO usar no próximo treino — a
    suposição original é que as duas visões são condicionalmente independentes dado o rótulo, o que
    reduz a chance de os dois cometerem o mesmo erro ao mesmo tempo.

    Pra imagens (sem uma divisão natural de features em "duas visões"), a adaptação usual é tratar
    duas arquiteturas/inicializações diferentes, ou dois pipelines de augmentation diferentes, como
    as duas "visões" — por isso model_a/model_b e unlabeled_loader_a/unlabeled_loader_b são
    independentes (podem ser o mesmo dataset com augmentations diferentes, ou visões genuinamente
    distintas). Os dois loaders devem iterar na MESMA ordem de unlabeled_indices.

    Devolve (pseudo_labels_para_a, pseudo_labels_para_b): cada um no formato (selected_indices,
    labels) de pseudo_labeling_strategy — a lista pra "a" vem das predições confiantes de b (pra
    treinar a com o que b rotulou), e vice-versa.
    """
    conf_a, preds_a = _confident_predictions(model_a, device, unlabeled_loader_a)
    conf_b, preds_b = _confident_predictions(model_b, device, unlabeled_loader_b)

    indices_for_a, labels_for_a = [], []  # rotulado por B, pra treinar A
    indices_for_b, labels_for_b = [], []  # rotulado por A, pra treinar B
    for i, idx in enumerate(unlabeled_indices):
        if conf_a[i].item() >= confidence_threshold:
            indices_for_b.append(idx)
            labels_for_b.append(preds_a[i])
        if conf_b[i].item() >= confidence_threshold:
            indices_for_a.append(idx)
            labels_for_a.append(preds_b[i])

    return ((np.array(indices_for_a, dtype=int), labels_for_a),
            (np.array(indices_for_b, dtype=int), labels_for_b))


def tri_training_strategy(models, device, unlabeled_loader, unlabeled_indices,
                           labeled_loader, prev_errors=None, prev_sizes=None, random_state=42, **kwargs):
    """
    Tri-Training (Zhou & Li, 2005, IEEE TKDE, "Tri-training: Exploiting Unlabeled Data Using Three
    Classifiers"): 3 classificadores, cada um idealmente treinado numa amostra bootstrap diferente do
    rotulado. A cada rodada, um exemplo não rotulado entra no conjunto de treino do classificador h_i
    SE OS OUTROS DOIS (h_j, h_k) concordam na predição — a concordância de dois classificadores
    independentes serve como "voto de confiança" sem precisar de threshold de probabilidade.

    Segue o critério prático de aceitação do paper original, pra evitar aceitar dados demais quando a
    taxa de erro estimada do par (h_j,h_k) não compensa: estima e_i, a taxa de erro combinada de
    (h_j,h_k) sobre labeled_loader (fração das vezes que os dois concordam E erram); só aceita os
    novos rótulos pra h_i se e_i for menor que o e_i aceito na rodada anterior — e, se o novo conjunto
    de candidatos for grande demais pro erro estimado, subamostra pra manter
    e_i·|L_i| < prev_e_i·|prev_L_i| (o "orçamento" esperado de rótulos errados introduzidos não
    cresce a cada rodada, o que fundamenta teoricamente a convergência do método no paper original).

    models: lista de 3 modelos [h1, h2, h3]. prev_errors/prev_sizes: listas de 3 valores — o e_i e
    |L_i| aceitos na rodada ANTERIOR (None na primeira rodada, quando tudo que passar no critério de
    concordância é aceito). labeled_loader: usado só pra estimar e_i (precisa de rótulo verdadeiro).

    Devolve uma lista de 3 tuplas (selected_indices, pseudo_labels, error_i, size_i) — uma por
    classificador; error_i/size_i devem ser passados como prev_errors[i]/prev_sizes[i] na próxima
    chamada, pra manter o critério de aceitação consistente entre rodadas.
    """
    def _hard_predictions(model, loader):
        model.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                images = batch[0].to(device)
                p = torch.argmax(model(images), dim=-1)
                preds.append(p.cpu())
        return torch.cat(preds)

    def _estimate_pair_error(model_j, model_k, loader):
        model_j.eval()
        model_k.eval()
        agree_total, agree_wrong = 0, 0
        with torch.no_grad():
            for batch in loader:
                images, targets = batch[0].to(device), batch[1].to(device)
                pred_j = torch.argmax(model_j(images), dim=-1)
                pred_k = torch.argmax(model_k(images), dim=-1)
                agree = pred_j == pred_k
                agree_total += agree.sum().item()
                agree_wrong += ((pred_j != targets) & agree).sum().item()
        return agree_wrong / agree_total if agree_total > 0 else 0.5

    all_preds = [_hard_predictions(m, unlabeled_loader) for m in models]
    unlabeled_indices = np.asarray(unlabeled_indices)
    rng = np.random.default_rng(random_state)

    results = []
    for i in range(3):
        j, k = [x for x in range(3) if x != i]
        error_i = _estimate_pair_error(models[j], models[k], labeled_loader)

        prev_error = prev_errors[i] if prev_errors is not None else 0.5
        prev_size = prev_sizes[i] if prev_sizes is not None else 0

        agree_mask = (all_preds[j] == all_preds[k]).numpy()
        candidate_positions = np.where(agree_mask)[0]

        if error_i >= prev_error or len(candidate_positions) == 0:
            # taxa de erro do par não melhorou: não atualiza h_i nessa rodada
            results.append((np.array([], dtype=int), [], prev_error, prev_size))
            continue

        size_i = len(candidate_positions)
        if prev_size > 0 and error_i * size_i >= prev_error * prev_size:
            # conjunto grande demais pro erro estimado: subamostra pra manter o "orçamento" de ruído
            max_size = int((prev_error * prev_size) / max(error_i, 1e-8)) - 1
            max_size = max(max_size, 0)
            if max_size < size_i:
                candidate_positions = rng.choice(candidate_positions, size=max_size, replace=False)
                size_i = len(candidate_positions)

        if size_i == 0:
            results.append((np.array([], dtype=int), [], error_i, 0))
            continue

        candidate_indices = unlabeled_indices[candidate_positions].astype(int)
        candidate_labels = [all_preds[j][p] for p in candidate_positions]
        results.append((candidate_indices, candidate_labels, error_i, size_i))

    return results


class PseudoLabeledDataset(Dataset):
    """
    Dataset dedicado a amostras nunca rotuladas: para cada índice, pega a imagem do dataset base e
    devolve o pseudo-rótulo previsto pelo modelo no lugar do rótulo — o rótulo real do dataset base
    é descartado sem nunca ser usado. Mantém os eventuais elementos extras do item original (ex.:
    o "length" do SVHNCustomDataset) pra ter a mesma aridade do conjunto rotulado de verdade, já que
    os dois são combinados num único DataLoader via ConcatDataset (misturar tuplas de tamanhos
    diferentes no mesmo batch quebraria o collate do PyTorch). Pseudo-rotulação nunca sobrescreve ou
    observa o rótulo de uma amostra já rotulada por um oráculo.
    """

    def __init__(self, base_dataset, indices, pseudo_labels):
        self.base_dataset = base_dataset
        self.indices = indices
        self.pseudo_labels = pseudo_labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        item = self.base_dataset[self.indices[i]]
        return (item[0], self.pseudo_labels[i]) + tuple(item[2:])