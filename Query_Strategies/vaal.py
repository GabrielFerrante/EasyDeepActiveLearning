import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# VAAL — Variational Adversarial Active Learning (Sinha, Ebrahimi & Darrell, 2019, ICCV,
# "Variational Adversarial Active Learning"): diferente das outras Query_Strategies, não usa o
# modelo alvo (nem suas predições nem seu get_embedding) — treina um VAE + discriminador adversarial
# à parte, só a partir das IMAGENS (rotuladas e não rotuladas), sem olhar rótulo nenhum. A ideia: o
# VAE aprende um espaço latente comum pras duas populações; o discriminador aprende a distinguir
# "veio do pool rotulado" de "veio do pool não rotulado"; e as amostras que o discriminador tem mais
# certeza de que são não rotuladas são as que representam melhor o que falta no conjunto rotulado.


def _kl_divergence(mu, logvar):
    """KL(N(mu,logvar) || N(0,I)), forma fechada padrão de VAE."""
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))


class VAEEncoder(nn.Module):
    def __init__(self, latent_dim=32, in_channels=3, base_channels=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4), nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.fc_mu = nn.Linear(base_channels * 8 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(base_channels * 8 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.pool(self.conv(x)).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=32, out_channels=3, base_channels=32):
        super().__init__()
        self.base_channels = base_channels
        self.fc = nn.Linear(latent_dim, base_channels * 8 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z, output_size):
        h = self.fc(z).view(-1, self.base_channels * 8, 4, 4)
        h = self.deconv(h)
        # As transposed-convs não garantem bater exatamente o tamanho de entrada; ajusta pra caber.
        return F.interpolate(h, size=output_size, mode="bilinear", align_corners=False)


class VAE(nn.Module):
    """β-VAE (Sec. 3.1 do paper). Arquitetura independente do modelo alvo — só enxerga as imagens."""

    def __init__(self, latent_dim=32, in_channels=3, base_channels=32):
        super().__init__()
        self.encoder = VAEEncoder(latent_dim, in_channels, base_channels)
        self.decoder = VAEDecoder(latent_dim, in_channels, base_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z, output_size=x.shape[-2:])
        return x_hat, mu, logvar, z


class Discriminator(nn.Module):
    """MLP de 5 camadas (conforme "Implementation details" do paper) que tenta prever, a partir do
    z do VAE, se a amostra veio do pool rotulado (1) ou não rotulado (0)."""

    def __init__(self, latent_dim=32, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


def train_vaal(vae, discriminator, labeled_loader, unlabeled_loader, vae_optimizer, disc_optimizer,
                device, epochs, beta=1.0, lambda1=1.0, lambda2=1.0):
    """
    Algorithm 1 do paper. A cada mini-batch, pareia uma amostra do pool rotulado com uma do pool não
    rotulado (o menor dos dois loaders manda o número de iterações por época; o outro é reciclado em
    loop). Duas atualizações por passo:

    1. VAE minimiza: λ1·(reconstrução + β·KL, nos dois pools) + λ2·(engana o discriminador, fazendo
       D(z_L) e D(z_U) irem pra 1 — i.e., tenta fazer o não rotulado parecer rotulado no espaço latente).
    2. Discriminador minimiza: cross-entropy binária padrão (D(z_L)->1, D(z_U)->0), usando os MESMOS
       z's mas sem deixar gradiente voltar pro encoder (senão a atualização do VAE acima seria desfeita).

    O modelo alvo da tarefa não participa desse treino — VAAL é inteiramente sobre as imagens.
    """
    for epoch in range(epochs):
        vae.train()
        discriminator.train()

        unlabeled_iter = iter(unlabeled_loader)
        progress_bar = tqdm(labeled_loader, desc=f"VAAL [{epoch + 1}/{epochs}]", unit="batch", leave=False)
        for labeled_batch in progress_bar:
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)

            x_l = labeled_batch[0].to(device)
            x_u = unlabeled_batch[0].to(device)

            # --- 1) Atualiza o VAE ---
            vae_optimizer.zero_grad()
            recon_l, mu_l, logvar_l, z_l = vae(x_l)
            recon_u, mu_u, logvar_u, z_u = vae(x_u)

            recon_loss = F.mse_loss(recon_l, x_l) + F.mse_loss(recon_u, x_u)
            kl_loss = _kl_divergence(mu_l, logvar_l) + _kl_divergence(mu_u, logvar_u)
            trd_loss = recon_loss + beta * kl_loss

            d_l = discriminator(z_l)
            d_u = discriminator(z_u)
            adv_loss = (F.binary_cross_entropy(d_l, torch.ones_like(d_l))
                        + F.binary_cross_entropy(d_u, torch.ones_like(d_u)))

            vae_loss = lambda1 * trd_loss + lambda2 * adv_loss
            vae_loss.backward()
            vae_optimizer.step()

            # --- 2) Atualiza o discriminador (com z's recomputados, sem gradiente pro encoder) ---
            disc_optimizer.zero_grad()
            with torch.no_grad():
                _, _, _, z_l_det = vae(x_l)
                _, _, _, z_u_det = vae(x_u)

            d_l_det = discriminator(z_l_det)
            d_u_det = discriminator(z_u_det)
            disc_loss = (F.binary_cross_entropy(d_l_det, torch.ones_like(d_l_det))
                         + F.binary_cross_entropy(d_u_det, torch.zeros_like(d_u_det)))
            disc_loss.backward()
            disc_optimizer.step()

            progress_bar.set_postfix(vae_loss=f"{vae_loss.item():.3f}", disc_loss=f"{disc_loss.item():.3f}")


def vaal_query_strategy(vae, discriminator, device, budget, unlabeled_loader, unlabeled_indices, **kwargs):
    """
    Algorithm 2 do paper: seleciona as amostras com MENOR D(z) — o discriminador está mais confiante
    de que elas vieram do pool NÃO rotulado, ou seja, são as que o espaço latente atual do VAE ainda
    não consegue confundir com o pool rotulado (mais informativas/diversas em relação ao que já foi
    rotulado).

    Não usa o modelo alvo da tarefa; vae/discriminator devem já estar treinados via train_vaal. Para
    plugar num ciclo de AL, amarre os módulos treinados antes de passar como query_strategy, ex.:

        query_strategy = lambda **kw: vaal_query_strategy(vae=trained_vae, discriminator=trained_disc, **kw)
    """
    vae.eval()
    discriminator.eval()
    all_scores = []

    print(f"Calculando score do discriminador (VAAL) para {len(unlabeled_indices)} amostras...")
    with torch.no_grad():
        for batch in tqdm(unlabeled_loader):
            images = batch[0].to(device)
            mu, logvar = vae.encoder(images)
            z = vae.reparameterize(mu, logvar)
            all_scores.extend(discriminator(z).cpu().numpy())

    all_scores = np.array(all_scores)
    selected_positions = np.argsort(all_scores)[:budget]
    return np.array(unlabeled_indices)[selected_positions]
