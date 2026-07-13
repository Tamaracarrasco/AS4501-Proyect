"""Semi-supervised multi-resolution VAE with a dedicated latent coordinate for redshift.

Input:
    X  : (N, 5, 4, 30, 30) from file_out_data/vae_input.npz
    z  : redshift target from the same NPZ, normalized on the training split.

Design:
    - The latent vector is split into one supervised coordinate z_sup and the
      remaining stochastic VAE coordinates z_vae.
    - z_sup is trained with an auxiliary regression loss.
    - KL is applied only to z_vae so the supervised coordinate is not pushed to
      N(0, I) and can carry redshift information.
    - The decoder receives [z_sup, z_vae] so the supervised coordinate remains
      part of the generative path.

This file is intentionally self-contained and reuses the same multi-resolution
encoder/decoder style as scripts/VAE.py, but with the semi-supervised latent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        peak_signal_noise_ratio as _psnr,
        structural_similarity_index_measure as _ssim,
    )
    HAS_TM = True
except Exception:
    HAS_TM = False


_REPO = Path(__file__).resolve().parent.parent

CONFIG = {
    "data_path": os.environ.get("VAE_DATA", str(_REPO / "file_out_data" / "vae_input.npz")),
    "out_dir": os.environ.get("VAE_OUT", str(_REPO / "file_out_data")),
    "run_id": None,
    "z_dim": 64,
    "z_sup_dim": 1,
    "branch_dim": 128,
    "fusion_dim": 256,
    "base_ch": 64,
    "share_level_weights": True,
    "beta_final": 1.0,
    "warmup_epochs": 30,
    "z_reg_weight": 1.0,
    "active_kl_thresh": 0.01,
    "metrics_every": 10,
    "lr": 1e-3,
    "batch_size": 128,
    "epochs": 200,
    "grad_clip": 5.0,
    "patience": 25,
    "sched_patience": 8,
    "seed": 42,
    "fig_every": 50,
    "viz_level": 2,
    "viz_mode": "rgb",
    "viz_band": 1,
    "n_viz": 8,
    "use_wandb": False,
    "wandb_project": "cosmos-vae",
    "wandb_run": None,
    "data_augmentation": False
}

BAND_NAMES = ["g", "r", "i", "z"]
N_LEVELS = 5
N_BANDS = 4
IMG = 30


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GPUBatches:
    """Mini-batch iterator for tensors already moved to the target device."""

    def __init__(self, X, z, batch_size, device, shuffle, seed=42, drop_last=False):
        self.X = X.to(device)
        self.z = z.to(device)
        self.bs = batch_size
        self.shuffle = shuffle
        self.device = device
        self.drop_last = drop_last
        self.g = torch.Generator().manual_seed(seed)

    def __len__(self):
        n = self.X.shape[0]
        return n // self.bs if self.drop_last else (n + self.bs - 1) // self.bs

    def __iter__(self):
        n = self.X.shape[0]
        idx = torch.randperm(n, generator=self.g) if self.shuffle else torch.arange(n)
        idx = idx.to(self.device)
        last = n - (n % self.bs) if self.drop_last else n
        for i in range(0, last, self.bs):
            sel = idx[i:i + self.bs]
            yield self.X[sel], self.z[sel], sel


def load_data(config, device):
    d = np.load(config["data_path"], allow_pickle=True)
    X = d["X"]
    tr, va = d["idx_train"], d["idx_val"]
    te = d["idx_test"] if "idx_test" in d else np.array([], dtype=int)
    types = d["type"]
    zred = np.asarray(d["z"], dtype=np.float32)
    bands = [str(b) for b in d["bands"]] if "bands" in d else BAND_NAMES

    z_mean = float(zred[tr].mean())
    z_std = float(zred[tr].std()) if float(zred[tr].std()) > 0 else 1.0
    z_norm = (zred - z_mean) / z_std

    Xtr = torch.as_tensor(X[tr], dtype=torch.float32).to(device)
    Xva = torch.as_tensor(X[va], dtype=torch.float32).to(device)
    ztr = torch.as_tensor(z_norm[tr], dtype=torch.float32).unsqueeze(1).to(device)
    zva = torch.as_tensor(z_norm[va], dtype=torch.float32).unsqueeze(1).to(device)

    train_loader = GPUBatches(Xtr, ztr, config["batch_size"], device, shuffle=True, seed=config["seed"], drop_last=True)
    val_loader = GPUBatches(Xva, zva, config["batch_size"], device, shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "X_val": Xva,
        "z_val": zva,
        "z_val_raw": zred[va].astype(np.float32),
        "type_val": np.asarray(types)[va],
        "n_train": len(tr),
        "n_val": len(va),
        "n_test": len(te),
        "x_shape": tuple(int(s) for s in X.shape),
        "bands": bands,
        "z_mean": z_mean,
        "z_std": z_std,
    }

def data_augmentation(x):
    """Aplicamos rotaciones en múltiplos de 90 grados y reflexiones aleatorias a un batch de tensores"""

    # Número de rotaciones (0, 1, 2, 3) que corresponden a 0°, 90°, 180° y 270°
    k = torch.randint(0, 4, (x.size(0),), device=x.device)

    # Aplicamos la rotación a cada imagen en el batch
    x_rot = torch.stack([torch.rot90(x[i], int(k[i]), dims=[-2, -1]) for i in range(x.size(0))])

    # Asignamos una probabilidad de 0.5 para reflejar horizontalmente
    flip_idx = torch.rand(x.size(0), device=x.device) < 0.5

    if flip_idx.any():
        x_rot[flip_idx] = torch.flip(x_rot[flip_idx], dims=[-1])  # Reflejo horizontal
    
    return x_rot

class EncoderBranch(nn.Module):
    def __init__(self, base_ch, branch_dim):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4
        self.conv = nn.Sequential(
            nn.Conv2d(N_BANDS, c1, 3, stride=2, padding=1),
            nn.BatchNorm2d(c1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),
            nn.BatchNorm2d(c2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1),
            nn.BatchNorm2d(c3), nn.LeakyReLU(0.2, inplace=True),
        )
        self.flat_dim = c3 * 4 * 4
        self.fc = nn.Linear(self.flat_dim, branch_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.flatten(1)
        return F.leaky_relu(self.fc(h), 0.2)


class DecoderBranch(nn.Module):
    def __init__(self, base_ch, branch_dim):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4
        self.c3 = c3
        self.fc = nn.Linear(branch_dim, c3 * 4 * 4)

        def ps_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout * 4, 3, padding=1), nn.PixelShuffle(2),
                nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True),
            )

        self.b1 = ps_block(c3, c2)
        self.b2 = ps_block(c2, c1)
        self.b3 = ps_block(c1, c1)
        self.out = nn.Conv2d(c1, N_BANDS, 3, padding=1)

    def forward(self, v):
        h = F.leaky_relu(self.fc(v), 0.2).view(-1, self.c3, 4, 4)
        h = self.b3(self.b2(self.b1(h)))
        h = h[:, :, 1:31, 1:31]
        return self.out(h)


class SemiSupervVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.L = N_LEVELS
        self.share = config["share_level_weights"]
        self.z_sup_dim = int(config["z_sup_dim"])
        self.z_vae_dim = int(config["z_dim"]) - self.z_sup_dim
        if self.z_vae_dim <= 0:
            raise ValueError("z_dim debe ser mayor que z_sup_dim")

        bd, bch = config["branch_dim"], config["base_ch"]
        n = 1 if self.share else self.L
        self.enc_branches = nn.ModuleList([EncoderBranch(bch, bd) for _ in range(n)])
        self.dec_branches = nn.ModuleList([DecoderBranch(bch, bd) for _ in range(n)])

        self.enc_fuse = nn.Sequential(
            nn.Linear(self.L * bd, config["fusion_dim"]),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc_mu = nn.Linear(config["fusion_dim"], self.z_vae_dim)
        self.fc_logvar = nn.Linear(config["fusion_dim"], self.z_vae_dim)
        self.fc_zsup = nn.Linear(config["fusion_dim"], self.z_sup_dim)

        self.dec_fuse = nn.Sequential(
            nn.Linear(config["z_dim"], config["fusion_dim"]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(config["fusion_dim"], self.L * bd),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.branch_dim = bd
        self.apply(self._init_weights)
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _enc(self, l):
        return self.enc_branches[0 if self.share else l]

    def _dec(self, l):
        return self.dec_branches[0 if self.share else l]

    def encode(self, x):
        B = x.size(0)
        if self.share:
            h = x.reshape(B * self.L, N_BANDS, IMG, IMG)
            h = self.enc_branches[0](h).view(B, self.L * self.branch_dim)
        else:
            h = torch.cat([self._enc(l)(x[:, l]) for l in range(self.L)], dim=1)
        h = self.enc_fuse(h)
        return self.fc_mu(h), self.fc_logvar(h), self.fc_zsup(h)

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        B = z.size(0)
        h = self.dec_fuse(z)
        if self.share:
            h = h.reshape(B * self.L, self.branch_dim)
            out = self.dec_branches[0](h).view(B, self.L, N_BANDS, IMG, IMG)
        else:
            chunks = h.view(B, self.L, self.branch_dim)
            outs = [self._dec(l)(chunks[:, l]) for l in range(self.L)]
            out = torch.stack(outs, dim=1)
        return out

    def forward(self, x):
        mu, logvar, z_sup = self.encode(x)
        z_vae = self.reparameterize(mu, logvar)
        z = torch.cat([z_sup, z_vae], dim=1)
        return self.decode(z), mu, logvar, z_sup


def semisuperv_loss(recon, x, mu, logvar, z_sup, z_target, beta, z_weight):
    recon_per_img = F.mse_loss(recon, x, reduction="none").sum(dim=[1, 2, 3, 4])
    recon_term = recon_per_img.mean()
    kl_per_img = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_term = kl_per_img.mean()
    z_loss = F.mse_loss(z_sup, z_target)
    total = recon_term + beta * kl_term + z_weight * z_loss
    return total, recon_term, kl_term, z_loss


def beta_schedule(epoch, config):
    w = max(1, config["warmup_epochs"])
    return config["beta_final"] * min(1.0, epoch / w)


def kl_per_dim(mu, logvar):
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(dim=0)


def train_epoch(model, loader, opt, beta, z_weight, device, grad_clip, data_aug_enabled=False):
    model.train()
    tot = rec = kl = zloss = 0.0
    n = 0
    for x, z, _ in loader:
        x = x.to(device)
        z = z.to(device)
        if data_aug_enabled:
            x = data_augmentation(x)  # Aplicamos data augmentation al batch de entrada
        opt.zero_grad()
        recon, mu, logvar, z_sup = model(x)
        loss, r, k, zl = semisuperv_loss(recon, x, mu, logvar, z_sup, z, beta, z_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        bs = x.size(0)
        tot += loss.item() * bs
        rec += r.item() * bs
        kl += k.item() * bs
        zloss += zl.item() * bs
        n += bs
    return {"total": tot / n, "recon": rec / n, "kl": kl / n, "z_loss": zloss / n}


@torch.no_grad()
def validate(model, loader, device, config, image_metrics=True):
    model.eval()
    beta_f = config["beta_final"]
    z_weight = config["z_reg_weight"]
    tot = rec = kl = zloss = mse_px = 0.0
    n = 0
    ssim_sum = psnr_sum = 0.0
    kld_accum = torch.zeros(config["z_dim"] - config["z_sup_dim"], device=device)
    z_pred = []
    z_true = []
    for x, z, _ in loader:
        x = x.to(device)
        z = z.to(device)
        recon, mu, logvar, z_sup = model(x)
        loss, r, k, zl = semisuperv_loss(recon, x, mu, logvar, z_sup, z, beta_f, z_weight)
        bs = x.size(0)
        tot += loss.item() * bs
        rec += r.item() * bs
        kl += k.item() * bs
        zloss += zl.item() * bs
        mse_px += F.mse_loss(recon, x, reduction="mean").item() * bs
        kld_accum += kl_per_dim(mu, logvar) * bs
        z_pred.append(z_sup.detach().cpu().numpy())
        z_true.append(z.detach().cpu().numpy())
        n += bs
        if HAS_TM and image_metrics:
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
    z_pred = np.concatenate(z_pred, axis=0).reshape(-1)
    z_true = np.concatenate(z_true, axis=0).reshape(-1)
    z_rmse = float(np.sqrt(np.mean((z_pred - z_true) ** 2)))
    z_corr = float(np.corrcoef(z_pred, z_true)[0, 1]) if np.std(z_pred) > 0 and np.std(z_true) > 0 else float("nan")
    out = {
        "total": tot / n,
        "recon": rec / n,
        "kl": kl / n,
        "z_loss": zloss / n,
        "mse_px": mse_px / n,
        "kl_per_dim_mean": float(kld.mean()),
        "active_dims": active,
        "kld_vec": kld,
        "z_rmse": z_rmse,
        "z_corr": z_corr,
    }
    if HAS_TM and image_metrics:
        out["ssim"] = ssim_sum / n
        out["psnr"] = psnr_sum / n
    return out


def _to_display(arr4, mode, band):
    def stretch(im):
        lo, hi = np.percentile(im, [1, 99])
        return np.clip((im - lo) / (hi - lo + 1e-8), 0, 1)
    if mode == "rgb":
        return np.stack([stretch(arr4[2]), stretch(arr4[1]), stretch(arr4[0])], axis=-1)
    return stretch(arr4[band])


def save_recon_grid(model, X_val, device, config, epoch, logger):
    model.eval()
    n = min(config["n_viz"], X_val.size(0))
    lvl = config["viz_level"]
    x = X_val[:n].to(device)
    with torch.no_grad():
        recon, _, _, _ = model(x)
    x = x.cpu().numpy()
    recon = recon.cpu().numpy()
    cmap = None if config["viz_mode"] == "rgb" else "gray"
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.4))
    axes = np.atleast_2d(axes)
    for j in range(n):
        axes[0, j].imshow(_to_display(x[j, lvl], config["viz_mode"], config["viz_band"]), origin="lower", cmap=cmap)
        axes[1, j].imshow(_to_display(recon[j, lvl], config["viz_mode"], config["viz_band"]), origin="lower", cmap=cmap)
        for r in (0, 1):
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
    axes[0, 0].set_ylabel("input", fontsize=11)
    axes[1, 0].set_ylabel("recon", fontsize=11)
    fig.suptitle(f"Reconstruccion (nivel {lvl}, {config['viz_mode']}) - epoca {epoch}")
    plt.tight_layout()
    logger.save_fig(fig, f"recon_e{epoch:04d}.png", wandb_key="recon")


def save_prior_samples(model, device, config, epoch, logger):
    model.eval()
    n = min(config["n_viz"], 8)
    lvl = config["viz_level"]
    with torch.no_grad():
        z = torch.randn(n, config["z_dim"], device=device)
        samp = model.decode(z).cpu().numpy()
    cmap = None if config["viz_mode"] == "rgb" else "gray"
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2.4))
    axes = np.atleast_1d(axes)
    for j in range(n):
        axes[j].imshow(_to_display(samp[j, lvl], config["viz_mode"], config["viz_band"]), origin="lower", cmap=cmap)
        axes[j].set_xticks([])
        axes[j].set_yticks([])
    fig.suptitle(f"Samples del prior z~N(0,I) (nivel {lvl}) - epoca {epoch}")
    plt.tight_layout()
    logger.save_fig(fig, f"prior_e{epoch:04d}.png", wandb_key="prior_samples")


@torch.no_grad()
def latent_umap(model, data, device, config, epoch, logger):
    model.eval()
    X_val = data["X_val"]
    mus = []
    z_sup = []
    z_true = []
    for i in range(0, X_val.size(0), config["batch_size"]):
        mu, _, zhat = model.encode(X_val[i:i + config["batch_size"]].to(device))
        mus.append(mu.cpu().numpy())
        z_sup.append(zhat.cpu().numpy())
    mu = np.concatenate(mus, axis=0)
    z_sup = np.concatenate(z_sup, axis=0).reshape(-1)
    z_true = data["z_val_raw"].reshape(-1)
    z_pred = z_sup * data["z_std"] + data["z_mean"]
    delta_z = (z_pred - z_true) / (1.0 + z_true)
    eta = float((np.abs(delta_z) > 0.05).mean() * 100.0)

    if HAS_UMAP:
        emb = umap.UMAP(n_components=2, random_state=config["seed"]).fit_transform(mu)
        method = "UMAP"
    else:
        from sklearn.decomposition import PCA
        emb = PCA(n_components=2).fit_transform(mu)
        method = "PCA (umap no instalado)"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))
    types = data["type_val"].astype(str)
    for t in np.unique(types):
        m = types == t
        axes[0].scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.5, label=t)
    axes[0].legend(markerscale=2, fontsize=8)
    axes[0].set_title("tipo morfologico")
    sc = axes[1].scatter(emb[:, 0], emb[:, 1], s=6, alpha=0.6, c=z_true, cmap="viridis")
    plt.colorbar(sc, ax=axes[1], label="z (redshift)")
    axes[1].set_title("redshift")
    axes[2].scatter(z_true, z_pred, s=8, alpha=0.6)
    lo = min(z_true.min(), z_pred.min())
    hi = max(z_true.max(), z_pred.max())
    axes[2].plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
    z_line = np.linspace(lo, hi, 200)
    threshold = 0.05
    axes[2].plot(z_line, z_line + threshold * (1 + z_line), color="red", lw=1.5, label=f"|dz| > {threshold}")
    axes[2].plot(z_line, z_line - threshold * (1 + z_line), color="red", lw=1.5)
    axes[2].set_xlabel("z real")
    axes[2].set_ylabel("z predicho por latent")
    axes[2].set_title("coordenada supervisada")
    axes[2].text(
        0.05,
        0.95,
        f"eta = {eta:.3f}%",
        transform=axes[2].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="none"),
    )
    axes[2].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Espacio latente ({method}, mu_vae) - epoca {epoch}")
    plt.tight_layout()
    logger.save_fig(fig, f"latent_e{epoch:04d}.png", wandb_key="latent_umap")


def resolve_run_id(config):
    if config["run_id"] is not None:
        return int(config["run_id"])
    used = []
    for p in Path(config["out_dir"]).glob("semi_superv_vae_summary*.json"):
        m = re.search(r"semi_superv_vae_summary(\d+)\.json$", p.name)
        if m:
            used.append(int(m.group(1)))
    return max(used) + 1 if used else 1


class Logger:
    def __init__(self, config):
        self.config = config
        rid = config["run_id"]
        out = Path(config["out_dir"])
        self.run_id = rid
        self.fig_dir = out / "figures" / f"semi_superv_vae{rid}"
        self.ckpt_dir = out / "checkpoints"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = out / f"semi_superv_vae_metrics{rid}.csv"
        self.summary_path = out / f"semi_superv_vae_summary{rid}.json"
        self.input_path = out / f"semi_superv_vae{rid}.input"
        self.ckpt_path = self.ckpt_dir / f"semi_superv_vae_best{rid}.pt"
        self.history = []
        self.use_wandb = config["use_wandb"] and HAS_WANDB
        if config["use_wandb"] and not HAS_WANDB:
            print("[logger] use_wandb=True pero wandb no esta instalado -> solo local.")
        if self.use_wandb:
            wandb.init(project=config["wandb_project"], name=config["wandb_run"], config=config)

    def write_input(self, config, data, model, device):
        n_par = sum(p.numel() for p in model.parameters())
        c1, c2, c3 = config["base_ch"], config["base_ch"] * 2, config["base_ch"] * 4
        bd, fd = config["branch_dim"], config["fusion_dim"]
        zd = config["z_dim"]
        zs = config["z_sup_dim"]
        shared = "compartida" if config["share_level_weights"] else "independiente"
        try:
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(_REPO), capture_output=True, text=True, timeout=5).stdout.strip() or "n/a"
        except Exception:
            commit = "n/a"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        arch = (
            f"[arquitectura] semi-supervised multi-res VAE (encoder {shared} entre {N_LEVELS} niveles)\n"
            f"  in ({N_LEVELS}x{N_BANDS}x{IMG}x{IMG})\n"
            f"   |- EncoderBranch x{N_LEVELS} [{shared}]: Conv {IMG}->15->8->4, ch {c1}->{c2}->{c3}, fc -> {bd}\n"
            f"        |- concat({N_LEVELS}x{bd}={N_LEVELS*bd}) -> fuse({fd})\n"
            f"        |- heads: mu_vae({zd-zs}), logvar_vae({zd-zs}), z_sup({zs})\n"
            f"  [latent split] z = [z_sup | z_vae] with z_sup supervised by redshift\n"
            f"  z({zd}) -> dec_fuse({fd}) -> {N_LEVELS}x{bd}\n"
            f"   |- DecoderBranch x{N_LEVELS} [{shared}]: fc->4x4({c3}), PixelShuffle 4->8->16->32, crop->{IMG}, conv -> {N_BANDS} bandas\n"
            f"  out ({N_LEVELS}x{N_BANDS}x{IMG}x{IMG})"
        )

        txt = f"""# semi_superv_vae{self.run_id}.input - reproducibility record
# generated: {ts} | run_id: {self.run_id}

[data]
data_path     : {config['data_path']}
X_shape       : {data['x_shape']}    # (N, levels, bands, H, W)
bands         : {', '.join(data['bands'])}
norm          : arcsinh(flux/sigma_bg) + zscore per channel (fit only on train)
z_target      : normalized using train mean/std
z_mean        : {data['z_mean']:.6f}
z_std         : {data['z_std']:.6f}
split         : train/val/test = {data['n_train']}/{data['n_val']}/{data['n_test']}  (stratified by type, seed {config['seed']})

[latent]
z_dim         : {zd}
z_sup_dim     : {zs}
z_vae_dim     : {zd - zs}

[hyperparameters]
seed          : {config['seed']}
base_ch       : {config['base_ch']}
branch_dim    : {bd}
fusion_dim    : {fd}
share_levels  : {config['share_level_weights']}
lr            : {config['lr']}
batch_size    : {config['batch_size']}
epochs        : {config['epochs']}   (early stopping patience={config['patience']})
grad_clip     : {config['grad_clip']}

[loss]
recon         : MSE gaussian (sum per image)
kl            : only on z_vae
z_reg         : MSE on z_sup vs normalized redshift
beta_final    : {config['beta_final']}
warmup_epochs : {config['warmup_epochs']}
z_reg_weight  : {config['z_reg_weight']}

[environment]
device        : {device.type}
torch         : {torch.__version__}
n_params      : {n_par/1e6:.2f}M
git_commit    : {commit}

{arch}
"""
        self.input_path.write_text(txt)
        print(f"[logger] written {self.input_path}")

    def log_scalars(self, epoch, d):
        row = {"epoch": epoch, **d}
        self.history.append(row)
        with open(self.metrics_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.history[0].keys()))
            w.writeheader()
            w.writerows(self.history)
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

    def save_loss_curves(self):
        if not self.history:
            return

        epochs = [row["epoch"] for row in self.history]

        recon_tr = [row["loss/recon_train"] for row in self.history]
        recon_va = [row["loss/recon_val"] for row in self.history]
        kl_tr = [row["loss/kl_train"] for row in self.history]
        kl_va = [row["loss/kl_val"] for row in self.history]
        z_tr = [row["loss/z_train"] for row in self.history]
        z_va = [row["loss/z_val"] for row in self.history]

        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        axes[0].plot(epochs, recon_tr, label="train", lw=2)
        axes[0].plot(epochs, recon_va, label="val", lw=2, ls="--")
        axes[0].set_ylabel("recon")
        axes[0].set_yscale("log")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(epochs, kl_tr, label="train", lw=2)
        axes[1].plot(epochs, kl_va, label="val", lw=2, ls="--")
        axes[1].set_ylabel("kl")
        axes[1].set_yscale("log")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        axes[2].plot(epochs, z_tr, label="train", lw=2)
        axes[2].plot(epochs, z_va, label="val", lw=2, ls="--")
        axes[2].set_ylabel("z_MSE")
        axes[2].set_xlabel("epoch")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        fig.suptitle("SemiSupervVAE losses per epoch")
        plt.tight_layout()
        self.save_fig(fig, "loss_curves.png")

def main(config=CONFIG, epoch_callback=None):
    set_seed(config["seed"])
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device: {device} | seed: {config['seed']} | umap: {HAS_UMAP} | wandb: {HAS_WANDB} | torchmetrics: {HAS_TM}")
    if device.type == "cpu":
        print("[aviso] sin GPU (CUDA/MPS) -> corriendo en CPU (lento).")

    data = load_data(config, device)
    print(f"train={data['n_train']}  val={data['n_val']}")

    model = SemiSupervVAE(config).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"modelo: {n_par/1e6:.2f}M params | share_level_weights={config['share_level_weights']}")

    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=config["sched_patience"])

    config = dict(config)
    config["run_id"] = resolve_run_id(config)
    print(f"run_id: {config['run_id']}")
    logger = Logger(config)
    logger.write_input(config, data, model, device)

    best_val = math.inf
    best_epoch = -1
    bad = 0
    ckpt = logger.ckpt_path

    for epoch in range(1, config["epochs"] + 1):
        beta = beta_schedule(epoch, config)
        tr = train_epoch(model, data["train_loader"], opt, beta, config["z_reg_weight"], device, config["grad_clip"], config["data_augmentation"])
        heavy = (epoch % config["metrics_every"] == 0)
        val = validate(model, data["val_loader"], device, config, image_metrics=heavy)
        sched.step(val["total"])
        lr_now = opt.param_groups[0]["lr"]

        logger.log_scalars(epoch, {
            "loss/total_train": tr["total"],
            "loss/recon_train": tr["recon"],
            "loss/kl_train": tr["kl"],
            "loss/z_train": tr["z_loss"],
            "loss/total_val": val["total"],
            "loss/recon_val": val["recon"],
            "loss/kl_val": val["kl"],
            "loss/z_val": val["z_loss"],
            "val/mse_px": val["mse_px"],
            "val/ssim": val.get("ssim", float("nan")),
            "val/psnr": val.get("psnr", float("nan")),
            "val/active_dims": val["active_dims"],
            "val/kl_per_dim_mean": val["kl_per_dim_mean"],
            "val/z_rmse": val["z_rmse"],
            "val/z_corr": val["z_corr"],
            "beta": beta,
            "lr": lr_now,
        })
        logger.save_loss_curves()
        print(
            f"e{epoch:03d} | beta={beta:.3f} | "
            f"train tot={tr['total']:.1f} rec={tr['recon']:.1f} kl={tr['kl']:.1f} z={tr['z_loss']:.3f} | "
            f"val tot={val['total']:.1f} rec={val['recon']:.1f} kl={val['kl']:.1f} z={val['z_loss']:.3f} | "
            f"z_rmse={val['z_rmse']:.3f} corr={val['z_corr']:.3f} act={val['active_dims']}/{config['z_dim'] - config['z_sup_dim']}"
        )

        if epoch % config["fig_every"] == 0:
            save_recon_grid(model, data["X_val"], device, config, epoch, logger)
            save_prior_samples(model, device, config, epoch, logger)
            latent_umap(model, data, device, config, epoch, logger)

        if epoch_callback is not None:
            epoch_callback(epoch, logger.history, model, data, device, config)

        if val["total"] < best_val - 1e-4:
            best_val = val["total"]
            best_epoch = epoch
            bad = 0
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch, "val": val["total"]}, ckpt)
        else:
            bad += 1
            if bad >= config["patience"]:
                print(f"early stopping en epoca {epoch} (mejor={best_epoch})")
                break

    print(f"mejor val={best_val:.2f} (epoca {best_epoch}); cargando checkpoint")
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    save_recon_grid(model, data["X_val"], device, config, best_epoch, logger)
    save_prior_samples(model, device, config, best_epoch, logger)
    latent_umap(model, data, device, config, best_epoch, logger)

    final = validate(model, data["val_loader"], device, config)
    summary = {
        "run_id": config["run_id"],
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "final_recon_val": final["recon"],
        "final_kl_val": final["kl"],
        "final_z_val": final["z_loss"],
        "final_z_rmse": final["z_rmse"],
        "final_z_corr": final["z_corr"],
        "final_mse_px": final["mse_px"],
        "final_active_dims": final["active_dims"],
        "final_ssim": final.get("ssim"),
        "final_psnr": final.get("psnr"),
        "z_dim": config["z_dim"],
        "z_sup_dim": config["z_sup_dim"],
        "out_files": {
            "metrics": logger.metrics_path.name,
            "input": logger.input_path.name,
            "figures": f"figures/semi_superv_vae{config['run_id']}/",
            "checkpoint": f"checkpoints/{logger.ckpt_path.name}",
        },
    }
    logger.summary_path.write_text(json.dumps(summary, indent=2))
    print("resumen:", json.dumps(summary, indent=2))
    logger.save_loss_curves()
    logger.finish()
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Semi-supervised VAE multi-resolution COSMOS")
    ap.add_argument("--data", help="ruta al vae_input.npz (default: $VAE_DATA o file_out_data/)")
    ap.add_argument("--out-dir", help="carpeta de salida (figuras, metricas, checkpoints)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--z-dim", type=int)
    ap.add_argument("--z-sup-dim", type=int)
    ap.add_argument("--beta-final", type=float)
    ap.add_argument("--z-reg-weight", type=float)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--run-id", type=int)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--no-share", action="store_true")
    ap.add_argument("--data-augmentation", action="store_true")
    args = ap.parse_args()

    cfg = dict(CONFIG)
    if args.data:
        cfg["data_path"] = args.data
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.z_dim:
        cfg["z_dim"] = args.z_dim
    if args.z_sup_dim is not None:
        cfg["z_sup_dim"] = args.z_sup_dim
    if args.beta_final is not None:
        cfg["beta_final"] = args.beta_final
    if args.z_reg_weight is not None:
        cfg["z_reg_weight"] = args.z_reg_weight
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.lr:
        cfg["lr"] = args.lr
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.run_id is not None:
        cfg["run_id"] = args.run_id
    if args.wandb:
        cfg["use_wandb"] = True
    if args.no_share:
        cfg["share_level_weights"] = False
    if args.data_augmentation:
        cfg["data_augmentation"] = True
    main(cfg)