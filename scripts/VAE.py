"""
VAE convolucional multi-resolución para stamps 

Input (producido por preprocess_vae.py):  X de shape (N, 5, 4, 30, 30)
    eje 1 -> 5 niveles de resolución (ramas)
    eje 2 -> 4 bandas g,r,i,z (canales de cada rama)
    datos ya normalizados: arcsinh(flux/sigma_bg) + z-score por canal.

Arquitectura tentativa:
    - 5 ramas, una por nivel de resolución. CADA RAMA COMPARTE LOS PESOS
      (un único encoder convolucional se aplica a los 5 niveles -> Siamese).
      Justificación: los 5 niveles son distintos FOV del MISMO objeto; un
      extractor de features compartido es eficiente en parámetros y fuerza una
      representación consistente de la estructura a través de las escalas.
      La interacción entre niveles ocurre en la capa de fusión.
    - Cada rama: Conv2d -> BatchNorm -> LeakyReLU, reduciendo 30->15->8->4.
    - Los 5 vectores de rama se concatenan y una MLP de fusión produce mu/logvar.
    - Decoder simétrico: z -> fusión inversa -> 5 vectores -> rama decoder
      compartida (upsample+conv) -> reconstrucción (N,5,4,30,30).

Decisiones de diseño no obvias (ver también comentarios inline):
    - Salida del decoder LINEAL (identidad): los datos están z-scoreados
      (no acotados, ~gaussianos) -> verosimilitud gaussiana -> pérdida MSE.
      (Si fueran [0,1] se usaría sigmoide + BCE.)
    - Recon = SUMA por imagen (no media) para que su escala sea comparable a la
      KL (que también es suma sobre dims latentes). Si se promediara por píxel,
      la KL aplastaría todo y habría posterior collapse.
    - KL annealing: beta sube 0 -> beta_final linealmente en las primeras
      `warmup_epochs`. Evita que al inicio la KL fuerce q(z|x)=N(0,I) (collapse).
    - Selección de modelo (early stopping / checkpoint / scheduler) usa la loss
      de validación evaluada SIEMPRE con beta_final (objetivo estacionario),
      aunque el entрenamiento use la beta annealed -> métrica comparable entre épocas.
    - upsample+conv en el decoder en vez de ConvTranspose para evitar artefactos
      de checkerboard.

Uso en Colab (extensión VSCode):
    !pip install umap-learn wandb        # umap requisito; wandb opcional
    from google.colab import drive; drive.mount('/content/drive')
    # editar CONFIG["data_path"] y CONFIG["out_dir"] a rutas de Drive/content
    # luego ejecutar este archivo (o llamar main(CONFIG)).

Local (CPU, env astro), prueba rápida:
    /opt/miniconda3/envs/astro/bin/python scripts/VAE.py
    (si umap no está instalado, la visualización latente cae a PCA 2D)
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# --- imports opcionales (Colab los tiene tras pip install; local pueden faltar) ---
try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

try:
    import wandb
    HAS_WANDB = True
except Exception:
    HAS_WANDB = False

try:
    from torchmetrics.functional import (
        structural_similarity_index_measure as _ssim,
        peak_signal_noise_ratio as _psnr,
    )
    HAS_TM = True
except Exception:
    HAS_TM = False


# =========================================================================== #
# CONFIG  — todos los hiperparámetros centralizados
# =========================================================================== #
_REPO = Path(__file__).resolve().parent.parent

CONFIG = {
    # datos / salida. Defaults reproducibles relativos al repo; se pueden
    # sobrescribir con $VAE_DATA / $VAE_OUT o con los flags --data / --out-dir.
    "data_path": os.environ.get("VAE_DATA", str(_REPO / "file_out_data" / "vae_input.npz")),
    "out_dir":   os.environ.get("VAE_OUT", str(_REPO / "file_out_data")),

    # arquitectura
    "z_dim":        64,        # dimensión latente (hiperparámetro)
    "branch_dim":   128,       # vector por rama (por nivel)
    "fusion_dim":   256,       # capa de fusión
    "base_ch":      32,        # canales base del encoder (32,64,128)
    "share_level_weights": True,   # True = un encoder compartido para los 5 niveles

    # pérdida
    "beta_final":   1.0,       # peso final de la KL
    "warmup_epochs": 30,       # épocas de warm-up lineal de beta (0 -> beta_final)
    "active_kl_thresh": 0.01,  # umbral KL/dim para contar dims "activas"

    # entrenamiento
    "lr":           1e-3,
    "batch_size":   128,
    "epochs":       200,
    "grad_clip":    5.0,
    "patience":     25,        # early stopping (épocas sin mejora)
    "sched_patience": 8,       # ReduceLROnPlateau
    "num_workers":  2,
    "seed":         42,

    # visualización / monitoreo
    "recon_every":  10,        # grilla recon + samples del prior cada N épocas
    "umap_every":   0,         # 0 = solo al final; >0 = cada K épocas
    "viz_level":    2,         # nivel a visualizar (2 = FOV 30")
    "viz_mode":     "rgb",     # "rgb" (g/r/i) o "band"
    "viz_band":     1,         # banda si viz_mode="band" (1 = r)
    "n_viz":        8,         # nº de ejemplos en las grillas

    # wandb (off por defecto; artefactos locales siempre se guardan)
    "use_wandb":     False,
    "wandb_project": "cosmos-vae",
    "wandb_run":     None,
}

BAND_NAMES = ["g", "r", "i", "z"]
N_LEVELS = 5
N_BANDS = 4
IMG = 30


# =========================================================================== #
# 0. Reproducibilidad
# =========================================================================== #
def set_seed(seed):
    """Fija TODAS las fuentes de aleatoriedad + flags deterministas de cuDNN.

    Garantiza que dos corridas en el MISMO hardware den el mismo resultado.
    (La equivalencia bit a bit entre GPUs distintas no está garantizada por
    cuDNN, pero el split, la init y el orden de batches sí son reproducibles.)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """Semilla por worker del DataLoader (cada uno deriva de la semilla global)."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


# =========================================================================== #
# 1. Dataset / dataloader
# =========================================================================== #
class StampDataset(Dataset):
    """Envuelve un tensor (N,5,4,30,30) ya normalizado. Devuelve (x, idx)."""

    def __init__(self, X):
        self.X = torch.as_tensor(X, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], i


def load_data(config):
    """Carga el .npz y arma loaders de train/val + arrays de val para UMAP.

    Devuelve dict con loaders, tensores de val, y etiquetas (type, z) de val.
    """
    d = np.load(config["data_path"], allow_pickle=True)
    X = d["X"]                                  # (N,5,4,30,30)
    tr, va = d["idx_train"], d["idx_val"]
    types = d["type"];  zred = d["z"]

    train_ds = StampDataset(X[tr])
    val_ds = StampDataset(X[va])

    g = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              shuffle=True, num_workers=config["num_workers"],
                              generator=g, worker_init_fn=seed_worker, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"],
                            shuffle=False, num_workers=config["num_workers"],
                            worker_init_fn=seed_worker)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "X_val": torch.as_tensor(X[va], dtype=torch.float32),
        "type_val": np.asarray(types)[va],
        "z_val": np.asarray(zred)[va].astype(float),
        "n_train": len(tr), "n_val": len(va),
    }


# =========================================================================== #
# 2. Modelo
# =========================================================================== #
class EncoderBranch(nn.Module):
    """CNN que mapea un nivel (4,30,30) -> vector branch_dim. 30->15->8->4."""

    def __init__(self, base_ch, branch_dim):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4
        self.conv = nn.Sequential(
            nn.Conv2d(N_BANDS, c1, 3, stride=2, padding=1),  # 30 -> 15
            nn.BatchNorm2d(c1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),       # 15 -> 8
            nn.BatchNorm2d(c2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1),       # 8 -> 4
            nn.BatchNorm2d(c3), nn.LeakyReLU(0.2, inplace=True),
        )
        self.flat_dim = c3 * 4 * 4
        self.fc = nn.Linear(self.flat_dim, branch_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.flatten(1)
        return F.leaky_relu(self.fc(h), 0.2)


class DecoderBranch(nn.Module):
    """Simétrico al encoder: vector branch_dim -> (4,30,30). 4->8->15->30."""

    def __init__(self, base_ch, branch_dim):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4
        self.c3 = c3
        self.fc = nn.Linear(branch_dim, c3 * 4 * 4)
        # upsample (interpolación) + conv para evitar checkerboard
        self.up1 = nn.Sequential(
            nn.Conv2d(c3, c2, 3, padding=1), nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.2, inplace=True))
        self.up2 = nn.Sequential(
            nn.Conv2d(c2, c1, 3, padding=1), nn.BatchNorm2d(c1),
            nn.LeakyReLU(0.2, inplace=True))
        # última conv -> 4 bandas, SIN activación (salida lineal, datos z-scoreados)
        self.out = nn.Conv2d(c1, N_BANDS, 3, padding=1)

    def forward(self, v):
        h = F.leaky_relu(self.fc(v), 0.2).view(-1, self.c3, 4, 4)
        h = F.interpolate(h, size=8, mode="bilinear", align_corners=False)
        h = self.up1(h)
        h = F.interpolate(h, size=15, mode="bilinear", align_corners=False)
        h = self.up2(h)
        h = F.interpolate(h, size=IMG, mode="bilinear", align_corners=False)
        return self.out(h)                                # (.,4,30,30) lineal


class MultiResVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.L = N_LEVELS
        self.share = config["share_level_weights"]
        bd, bch = config["branch_dim"], config["base_ch"]

        # ramas: 1 compartida, o 5 independientes
        n = 1 if self.share else self.L
        self.enc_branches = nn.ModuleList([EncoderBranch(bch, bd) for _ in range(n)])
        self.dec_branches = nn.ModuleList([DecoderBranch(bch, bd) for _ in range(n)])

        # fusión encoder: 5*branch_dim -> fusion_dim -> (mu, logvar)
        self.enc_fuse = nn.Sequential(
            nn.Linear(self.L * bd, config["fusion_dim"]),
            nn.LeakyReLU(0.2, inplace=True))
        self.fc_mu = nn.Linear(config["fusion_dim"], config["z_dim"])
        self.fc_logvar = nn.Linear(config["fusion_dim"], config["z_dim"])

        # fusión decoder: z -> fusion_dim -> 5*branch_dim
        self.dec_fuse = nn.Sequential(
            nn.Linear(config["z_dim"], config["fusion_dim"]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(config["fusion_dim"], self.L * bd),
            nn.LeakyReLU(0.2, inplace=True))
        self.branch_dim = bd
        self.apply(self._init_weights)
        # logvar head a cero -> al inicio logvar=0 -> std=1 (arranque estable)
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _enc(self, l):  # rama encoder para el nivel l
        return self.enc_branches[0 if self.share else l]

    def _dec(self, l):
        return self.dec_branches[0 if self.share else l]

    def encode(self, x):                                  # x: (B,5,4,30,30)
        B = x.size(0)
        if self.share:
            h = x.reshape(B * self.L, N_BANDS, IMG, IMG)
            h = self.enc_branches[0](h).view(B, self.L * self.branch_dim)
        else:
            h = torch.cat([self._enc(l)(x[:, l]) for l in range(self.L)], dim=1)
        h = self.enc_fuse(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu                                     # en eval usamos la media
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):                                  # -> (B,5,4,30,30)
        B = z.size(0)
        h = self.dec_fuse(z)                              # (B, 5*branch_dim)
        if self.share:
            h = h.reshape(B * self.L, self.branch_dim)
            out = self.dec_branches[0](h).view(B, self.L, N_BANDS, IMG, IMG)
        else:
            chunks = h.view(B, self.L, self.branch_dim)
            outs = [self._dec(l)(chunks[:, l]) for l in range(self.L)]
            out = torch.stack(outs, dim=1)
        return out

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# =========================================================================== #
# 3. Pérdida
# =========================================================================== #
def vae_loss(recon, x, mu, logvar, beta):
    """Devuelve (total, recon, kl) — recon y kl SEPARADOS.

    recon : MSE SUMADA por imagen, promediada en el batch.
    kl    : KL[q(z|x) || N(0,I)] en forma cerrada, sumada por imagen y promediada.
    """
    recon_per_img = F.mse_loss(recon, x, reduction="none").sum(dim=[1, 2, 3, 4])
    recon_term = recon_per_img.mean()

    kl_per_img = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_term = kl_per_img.mean()

    return recon_term + beta * kl_term, recon_term, kl_term


def beta_schedule(epoch, config):
    """Warm-up lineal de beta: 0 -> beta_final en `warmup_epochs`."""
    w = max(1, config["warmup_epochs"])
    return config["beta_final"] * min(1.0, epoch / w)


def kl_per_dim(mu, logvar):
    """KL promedio por dimensión latente (vector z_dim) -> diagnóstico de collapse."""
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(dim=0)


# =========================================================================== #
# 4. Train / validate
# =========================================================================== #
def train_epoch(model, loader, opt, beta, device, grad_clip):
    model.train()
    tot = rec = kl = 0.0
    n = 0
    for x, _ in loader:
        x = x.to(device)
        opt.zero_grad()
        recon, mu, logvar = model(x)
        loss, r, k = vae_loss(recon, x, mu, logvar, beta)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        bs = x.size(0)
        tot += loss.item() * bs; rec += r.item() * bs; kl += k.item() * bs; n += bs
    return {"total": tot / n, "recon": rec / n, "kl": kl / n}


@torch.no_grad()
def validate(model, loader, device, config):
    """Valida con beta_final (objetivo estacionario) y calcula diagnósticos."""
    model.eval()
    beta_f = config["beta_final"]
    tot = rec = kl = mse_px = 0.0
    n = 0
    ssim_sum = psnr_sum = 0.0
    kld_accum = torch.zeros(config["z_dim"], device=device)
    for x, _ in loader:
        x = x.to(device)
        recon, mu, logvar = model(x)
        loss, r, k = vae_loss(recon, x, mu, logvar, beta_f)
        bs = x.size(0)
        tot += loss.item() * bs; rec += r.item() * bs; kl += k.item() * bs
        mse_px += F.mse_loss(recon, x, reduction="mean").item() * bs
        kld_accum += kl_per_dim(mu, logvar) * bs
        n += bs
        # SSIM/PSNR sobre (B*5,4,30,30); data_range del propio batch
        if HAS_TM:
            xr = x.reshape(-1, N_BANDS, IMG, IMG)
            rr = recon.reshape(-1, N_BANDS, IMG, IMG)
            dr = float(xr.max() - xr.min())
            try:
                ssim_sum += _ssim(rr, xr, data_range=dr).item() * bs
                psnr_sum += _psnr(rr, xr, data_range=dr).item() * bs
            except Exception:
                pass

    kld = (kld_accum / n).cpu().numpy()
    active = int((kld > config["active_kl_thresh"]).sum())
    out = {"total": tot / n, "recon": rec / n, "kl": kl / n,
           "mse_px": mse_px / n, "kl_per_dim_mean": float(kld.mean()),
           "active_dims": active, "kld_vec": kld}
    if HAS_TM:
        out["ssim"] = ssim_sum / n
        out["psnr"] = psnr_sum / n
    return out


# =========================================================================== #
# 5. Visualización
# =========================================================================== #
def _to_display(arr4, mode, band):
    """arr4: (4,30,30) numpy -> imagen para mostrar (HxW o HxWx3), normalizada display."""
    def stretch(im):
        lo, hi = np.percentile(im, [1, 99])
        return np.clip((im - lo) / (hi - lo + 1e-8), 0, 1)
    if mode == "rgb":
        # convención astro: R=i, G=r, B=g  (índices 2,1,0)
        rgb = np.stack([stretch(arr4[2]), stretch(arr4[1]), stretch(arr4[0])], axis=-1)
        return rgb
    return stretch(arr4[band])


def save_recon_grid(model, X_val, device, config, epoch, logger):
    """Grilla input (arriba) vs reconstrucción (abajo), nivel y modo configurables."""
    model.eval()
    n = config["n_viz"]; lvl = config["viz_level"]
    x = X_val[:n].to(device)
    with torch.no_grad():
        recon, _, _ = model(x)
    x = x.cpu().numpy(); recon = recon.cpu().numpy()
    cmap = None if config["viz_mode"] == "rgb" else "gray"

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.4))
    for j in range(n):
        axes[0, j].imshow(_to_display(x[j, lvl], config["viz_mode"], config["viz_band"]),
                          origin="lower", cmap=cmap)
        axes[1, j].imshow(_to_display(recon[j, lvl], config["viz_mode"], config["viz_band"]),
                          origin="lower", cmap=cmap)
        for r in (0, 1):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
    axes[0, 0].set_ylabel("input", fontsize=11)
    axes[1, 0].set_ylabel("recon", fontsize=11)
    fig.suptitle(f"Reconstrucción (nivel {lvl}, {config['viz_mode']}) — época {epoch}")
    plt.tight_layout()
    logger.save_fig(fig, f"recon_e{epoch:04d}.png", wandb_key="recon")


def save_prior_samples(model, device, config, epoch, logger):
    """Decodifica z ~ N(0,I) para ver capacidad generativa (no solo recon)."""
    model.eval()
    n = config["n_viz"]; lvl = config["viz_level"]
    with torch.no_grad():
        z = torch.randn(n, config["z_dim"], device=device)
        samp = model.decode(z).cpu().numpy()
    cmap = None if config["viz_mode"] == "rgb" else "gray"
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.4))
    for j in range(n):
        axes[j].imshow(_to_display(samp[j, lvl], config["viz_mode"], config["viz_band"]),
                       origin="lower", cmap=cmap)
        axes[j].set_xticks([]); axes[j].set_yticks([])
    fig.suptitle(f"Samples del prior z~N(0,I) (nivel {lvl}) — época {epoch}")
    plt.tight_layout()
    logger.save_fig(fig, f"prior_e{epoch:04d}.png", wandb_key="prior_samples")


@torch.no_grad()
def latent_umap(model, data, device, config, epoch, logger):
    """Codifica todo val (usa mu), reduce a 2D (UMAP o PCA fallback), colorea."""
    model.eval()
    X_val = data["X_val"]
    mus = []
    for i in range(0, X_val.size(0), config["batch_size"]):
        mu, _ = model.encode(X_val[i:i + config["batch_size"]].to(device))
        mus.append(mu.cpu().numpy())
    mu = np.concatenate(mus, axis=0)

    if HAS_UMAP:
        emb = umap.UMAP(n_components=2, random_state=config["seed"]).fit_transform(mu)
        method = "UMAP"
    else:
        from sklearn.decomposition import PCA
        emb = PCA(n_components=2).fit_transform(mu)
        method = "PCA (umap no instalado)"

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # (a) coloreado por tipo morfológico (categórico)
    types = data["type_val"].astype(str)
    for t in np.unique(types):
        m = types == t
        axes[0].scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.5, label=t)
    axes[0].legend(markerscale=2, fontsize=9); axes[0].set_title("tipo morfológico")
    # (b) coloreado por redshift (continuo)
    sc = axes[1].scatter(emb[:, 0], emb[:, 1], s=6, alpha=0.6,
                         c=data["z_val"], cmap="viridis")
    plt.colorbar(sc, ax=axes[1], label="z (redshift)")
    axes[1].set_title("redshift")
    fig.suptitle(f"Espacio latente ({method}, mu) — época {epoch}")
    plt.tight_layout()
    logger.save_fig(fig, f"umap_e{epoch:04d}.png", wandb_key="latent_umap")


# =========================================================================== #
# 6. Logger  (local siempre; wandb opcional detrás de flag)
# =========================================================================== #
class Logger:
    def __init__(self, config):
        self.config = config
        self.fig_dir = Path(config["out_dir"]) / "figures" / "vae"
        self.ckpt_dir = Path(config["out_dir"]) / "checkpoints"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = Path(config["out_dir"]) / "vae_metrics.csv"
        self.history = []
        self.use_wandb = config["use_wandb"] and HAS_WANDB
        if config["use_wandb"] and not HAS_WANDB:
            print("[logger] use_wandb=True pero wandb no está instalado -> solo local.")
        if self.use_wandb:
            wandb.init(project=config["wandb_project"], name=config["wandb_run"],
                       config=config)

    def log_scalars(self, epoch, d):
        row = {"epoch": epoch, **d}
        self.history.append(row)
        # CSV incremental (lo ve el equipo vía git)
        import csv
        with open(self.metrics_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.history[0].keys()))
            w.writeheader(); w.writerows(self.history)
        if self.use_wandb:
            wandb.log({k: v for k, v in d.items()}, step=epoch)

    def save_fig(self, fig, name, wandb_key=None):
        path = self.fig_dir / name
        fig.savefig(path, dpi=110, bbox_inches="tight")
        if self.use_wandb and wandb_key:
            wandb.log({wandb_key: wandb.Image(str(path))})
        plt.close(fig)

    def finish(self):
        if self.use_wandb:
            wandb.finish()


# =========================================================================== #
# 7. Loop principal
# =========================================================================== #
def main(config=CONFIG, epoch_callback=None):
    """epoch_callback(epoch, history, model, data, device, config): hook opcional
    que se llama al final de cada época (p.ej. para graficar la loss en vivo en
    un notebook). Si es None, no hace nada (modo CLI normal)."""
    set_seed(config["seed"])
    # Device: CUDA (NVIDIA, p.ej. Colab) -> MPS (GPU Apple Silicon) -> CPU.
    # Opción anterior (solo CUDA), descomenta si MPS diera problemas con algún op:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")          # GPU de tu Mac (Apple Silicon)
    else:
        device = torch.device("cpu")
    print(f"device: {device} | seed: {config['seed']} | umap: {HAS_UMAP} | "
          f"wandb: {HAS_WANDB} | torchmetrics: {HAS_TM}")
    if device.type == "cpu":
        print("[aviso] sin GPU (CUDA/MPS) -> corriendo en CPU (lento).")

    data = load_data(config)
    print(f"train={data['n_train']}  val={data['n_val']}")

    model = MultiResVAE(config).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"modelo: {n_par/1e6:.2f}M params | share_level_weights={config['share_level_weights']}")

    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=config["sched_patience"])
    logger = Logger(config)

    best_val = math.inf; best_epoch = -1; bad = 0
    ckpt = logger.ckpt_dir / "vae_best.pt"

    for epoch in range(1, config["epochs"] + 1):
        beta = beta_schedule(epoch, config)
        tr = train_epoch(model, data["train_loader"], opt, beta, device,
                         config["grad_clip"])
        val = validate(model, data["val_loader"], device, config)
        sched.step(val["total"])
        lr_now = opt.param_groups[0]["lr"]

        # log escalares (recon y kl SIEMPRE separados)
        logger.log_scalars(epoch, {
            "loss/total_train": tr["total"], "loss/recon_train": tr["recon"],
            "loss/kl_train": tr["kl"],
            "loss/total_val": val["total"], "loss/recon_val": val["recon"],
            "loss/kl_val": val["kl"], "val/mse_px": val["mse_px"],
            "val/ssim": val.get("ssim", float("nan")),
            "val/psnr": val.get("psnr", float("nan")),
            "val/active_dims": val["active_dims"],
            "val/kl_per_dim_mean": val["kl_per_dim_mean"],
            "beta": beta, "lr": lr_now,
        })
        print(f"e{epoch:03d} | beta={beta:.3f} | "
              f"train tot={tr['total']:.1f} rec={tr['recon']:.1f} kl={tr['kl']:.1f} | "
              f"val tot={val['total']:.1f} rec={val['recon']:.1f} kl={val['kl']:.1f} | "
              f"act={val['active_dims']}/{config['z_dim']} "
              f"ssim={val.get('ssim', float('nan')):.3f}")

        # visualizaciones periódicas
        if epoch % config["recon_every"] == 0:
            save_recon_grid(model, data["X_val"], device, config, epoch, logger)
            save_prior_samples(model, device, config, epoch, logger)
        if config["umap_every"] and epoch % config["umap_every"] == 0:
            latent_umap(model, data, device, config, epoch, logger)

        # hook de monitoreo en vivo (notebook). Va después de guardar las figuras
        # para que el callback pueda mostrar las PNG recién generadas.
        if epoch_callback is not None:
            epoch_callback(epoch, logger.history, model, data, device, config)

        # checkpoint + early stopping (sobre val total con beta_final)
        if val["total"] < best_val - 1e-4:
            best_val = val["total"]; best_epoch = epoch; bad = 0
            torch.save({"model": model.state_dict(), "config": config,
                        "epoch": epoch, "val": val["total"]}, ckpt)
        else:
            bad += 1
            if bad >= config["patience"]:
                print(f"early stopping en época {epoch} (mejor={best_epoch})")
                break

    # restaura el mejor modelo y hace UMAP final
    print(f"mejor val={best_val:.2f} (época {best_epoch}); cargando checkpoint")
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    latent_umap(model, data, device, config, best_epoch, logger)
    save_recon_grid(model, data["X_val"], device, config, best_epoch, logger)

    # resumen final
    final = validate(model, data["val_loader"], device, config)
    summary = {"best_epoch": best_epoch, "best_val_total": best_val,
               "final_recon_val": final["recon"], "final_kl_val": final["kl"],
               "final_mse_px": final["mse_px"], "final_active_dims": final["active_dims"],
               "final_ssim": final.get("ssim"), "final_psnr": final.get("psnr"),
               "z_dim": config["z_dim"]}
    (Path(config["out_dir"]) / "vae_summary.json").write_text(json.dumps(summary, indent=2))
    print("resumen:", json.dumps(summary, indent=2))
    logger.finish()
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="VAE multi-resolución COSMOS")
    ap.add_argument("--data", help="ruta al vae_input.npz (default: $VAE_DATA o file_out_data/)")
    ap.add_argument("--out-dir", help="carpeta de salida (figuras, métricas, checkpoints)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--z-dim", type=int)
    ap.add_argument("--beta-final", type=float)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--wandb", action="store_true", help="activa logging a Weights & Biases")
    ap.add_argument("--no-share", action="store_true",
                    help="pesos independientes por nivel (en vez de compartidos)")
    args = ap.parse_args()

    cfg = dict(CONFIG)
    if args.data:        cfg["data_path"] = args.data
    if args.out_dir:     cfg["out_dir"] = args.out_dir
    if args.epochs:      cfg["epochs"] = args.epochs
    if args.z_dim:       cfg["z_dim"] = args.z_dim
    if args.beta_final is not None: cfg["beta_final"] = args.beta_final
    if args.batch_size:  cfg["batch_size"] = args.batch_size
    if args.lr:          cfg["lr"] = args.lr
    if args.seed is not None: cfg["seed"] = args.seed
    if args.wandb:       cfg["use_wandb"] = True
    if args.no_share:    cfg["share_level_weights"] = False
    main(cfg)
