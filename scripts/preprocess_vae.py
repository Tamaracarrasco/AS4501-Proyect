"""
Decisiones acordadas para el VAE:
  - Arquitectura multi-resolución POR RAMAS  -> se conserva el eje de nivel.
    El tensor de salida es (N, 5, 4, 30, 30): N objetos, 5 niveles (ramas),
    4 canales de banda cada uno.
  - Filtrado ESTRICTO de la muestra: maskbits==0, flujo>0, cortes de
    profundidad (magnitud corregida por extinción) y SNR>=3 en todas las bandas,
    más eliminación de stamps con píxeles NaN.
  - Normalización de intensidad: arcsinh + z-score POR CANAL (canal = banda x nivel).
    El arcsinh comprime el rango dinámico (preserva el fondo ~0); el z-score
    deja los 20 canales (4 bandas x 5 niveles) en escala comparable.
    Las estadísticas se estiman SOLO sobre el split de entrenamiento (sin fuga).

Pipeline:
  1. load_features  -> filtros de catálogo sobre el CSV de features.
  2. load_stamps    -> lee el tar, detecta NaN.
  3. align          -> intersección por desi_id y descarte de NaN.
  4. split_ids      -> train/val/test.
  5. fit_norm       -> softening (arcsinh) y mu/std por canal, en train.
  6. apply_norm     -> arcsinh + z-score.
  7. main           -> guarda .npz listo para el VAE.

Uso:
    # prueba rápida (carga solo N stamps):
    python scripts/preprocess_vae.py --n-max 300
    # corrida completa:
    python scripts/preprocess_vae.py
Ejecutar con el entorno `astro`:
    /opt/miniconda3/envs/astro/bin/python scripts/preprocess_vae.py
"""

import argparse
import io
import json
import os
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.stats import sigma_clipped_stats
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Rutas y constantes
# --------------------------------------------------------------------------- #
_REPO = Path(__file__).resolve().parent.parent

# Paths reproducibles para el equipo: por defecto el repo (datos en /data/, ver
# README). Se pueden sobrescribir con variables de entorno o los flags --tar/--features
# sin tocar el código (p.ej. si tienes el tar en otra carpeta):
#   export COSMOS_TAR="/ruta/a/cosmos_TC_202602.tar.gz.part_aa"
TAR_PATH = Path(os.environ.get("COSMOS_TAR", _REPO / "data" / "cosmos_TC_202602.tar.gz.part_aa"))
FEATURES_CSV = Path(os.environ.get("COSMOS_FEATURES", _REPO / "data" / "features_images_20260618.csv"))
OUT_DIR = _REPO / "file_out_data"

BANDS = ["g", "r", "i", "z"]            # orden del eje 0 del stamp
N_LEVELS = 5                             # eje 1
H = W = 30

# Coeficientes de extinción R_lambda para filtros DECam (Schlafly), usados en el EDA.
R_LAMBDA = {"g": 3.214, "r": 2.165, "i": 1.592, "z": 1.211}

# Cortes de profundidad sobre magnitud AB corregida por extinción (del EDA:
# Li Changhua et al. 2023 para g/r/z, DeROSITAS para i).
DEPTH_CUT = {"g": 24.0, "r": 23.4, "i": 23.3, "z": 22.5}

SNR_MIN = 3.0
RANDOM_SEED = 42
SPLIT_FRACS = (0.8, 0.1, 0.1)            # train, val, test


# --------------------------------------------------------------------------- #
# 1. Features + filtros de catálogo
# --------------------------------------------------------------------------- #
def load_features(path=FEATURES_CSV):
    """Carga el CSV y aplica los filtros de catálogo acordados.

    Devuelve un DataFrame con desi_id (str) ya filtrado.
    """
    df = pd.read_csv(path, sep=",")
    df["desi_id"] = df["desi_id"].astype(str)
    n0 = len(df)

    # FILTRO 1: calidad fotométrica.
    df = df[df["maskbits"] == 0].copy()
    n_mask = len(df)

    # FILTRO 2: flujo positivo en todas las bandas (necesario para magnitudes).
    pos = np.logical_and.reduce([df[f"flux_{b}"] > 0 for b in BANDS])
    df = df[pos].copy()
    n_pos = len(df)

    # Magnitud AB corregida por extinción galáctica:
    #   flux_corr = flux / mw_transmission   (mw_transmission ya está en unidades lineales)
    #   mag = 22.5 - 2.5*log10(flux_corr)
    for b in BANDS:
        flux_corr = df[f"flux_{b}"] / df[f"mw_transmission_{b}"]
        df[f"mag_{b}_corr"] = 22.5 - 2.5 * np.log10(flux_corr)

    # FILTRO 3: cortes de profundidad (descartar objetos demasiado tenues).
    depth_ok = np.logical_and.reduce(
        [df[f"mag_{b}_corr"] < DEPTH_CUT[b] for b in BANDS]
    )
    df = df[depth_ok].copy()
    n_depth = len(df)

    # FILTRO 4: SNR de catálogo = flux * sqrt(flux_ivar) >= 3 en todas las bandas.
    for b in BANDS:
        df[f"SNR_{b}"] = df[f"flux_{b}"] * np.sqrt(df[f"flux_ivar_{b}"])
    snr_ok = np.logical_and.reduce([df[f"SNR_{b}"] >= SNR_MIN for b in BANDS])
    df = df[snr_ok].copy()
    n_snr = len(df)

    print("Filtros de catálogo:")
    print(f"  inicial            : {n0}")
    print(f"  maskbits==0        : {n_mask}")
    print(f"  flujo>0 (griz)     : {n_pos}")
    print(f"  profundidad (mag)  : {n_depth}")
    print(f"  SNR>=3 (griz)      : {n_snr}")
    return df


# --------------------------------------------------------------------------- #
# 2. Stamps
# --------------------------------------------------------------------------- #
def load_stamps(path=TAR_PATH, keep_ids=None, n_max=None):
    """Lee los .npy del tar.

    keep_ids : set de desi_id a conservar (None = todos). Filtrar acá evita
               cargar en memoria stamps que el catálogo ya descartó.
    n_max    : tope de stamps a leer (para pruebas).

    Devuelve (stamps, names, n_nan) donde:
       stamps : dict desi_id -> array (4,5,30,30) float32  (solo sin NaN)
       names  : lista de ids leídos
       n_nan  : cuántos stamps se descartaron por contener NaN.
    """
    stamps, names, n_nan = {}, [], 0
    try:
        with tarfile.open(path, "r|gz") as tar:  # streaming, tolera truncamiento
            for m in tar:
                if not m.name.endswith(".npy"):
                    continue
                name = m.name.split("/")[-1].replace(".npy", "")
                if keep_ids is not None and name not in keep_ids:
                    continue
                f = tar.extractfile(m)
                if f is None:
                    continue
                arr = np.load(io.BytesIO(f.read())).astype(np.float32)
                names.append(name)
                if np.isnan(arr).any():           # descarte de stamps con NaN
                    n_nan += 1
                    continue
                stamps[name] = arr
                if n_max is not None and len(stamps) >= n_max:
                    break
    except (EOFError, tarfile.ReadError):
        print(f"  (tar truncado: recuperados {len(names)} stamps antes del corte)")
    return stamps, names, n_nan


# --------------------------------------------------------------------------- #
# 3. Alineación imagen <-> features
# --------------------------------------------------------------------------- #
def align(df, stamps):
    """Intersección por desi_id. Devuelve (ids, X, df_alineado).

    X tiene shape (N, 5, 4, 30, 30): se transpone (banda, nivel) -> (nivel, banda)
    para que cada nivel sea una rama con 4 canales de banda.
    """
    ids = [i for i in df["desi_id"] if i in stamps]
    X = np.stack([stamps[i] for i in ids])                 # (N,4,5,30,30)
    X = np.transpose(X, (0, 2, 1, 3, 4)).copy()            # (N,5,4,30,30)
    df_al = df.set_index("desi_id").loc[ids].reset_index()
    return ids, X, df_al


# --------------------------------------------------------------------------- #
# 4. Split
# --------------------------------------------------------------------------- #
def split_ids(n, stratify=None, fracs=SPLIT_FRACS, seed=RANDOM_SEED):
    """Índices para train/val/test usando sklearn.

    stratify : array de etiquetas (p.ej. tipo morfológico) para que los tres
               splits conserven la misma proporción de clases. None = aleatorio.

    Se hace en dos cortes: primero train vs resto, luego resto en val/test
    proporcional, porque train_test_split solo parte en dos.
    """
    f_tr, f_va, f_te = fracs
    idx = np.arange(n)

    # corte 1: train vs (val + test)
    tr, tmp = train_test_split(
        idx, train_size=f_tr, random_state=seed, stratify=stratify
    )

    # corte 2: val vs test, proporcional dentro de lo que quedó
    strat_tmp = stratify[tmp] if stratify is not None else None
    rel_val = f_va / (f_va + f_te)
    va, te = train_test_split(
        tmp, train_size=rel_val, random_state=seed, stratify=strat_tmp
    )
    return tr, va, te


# --------------------------------------------------------------------------- #
# 5. Normalización: arcsinh + z-score por canal (canal = nivel x banda)
# --------------------------------------------------------------------------- #
def fit_norm(X_train, max_stamps=2000, seed=RANDOM_SEED):
    """Estima, por canal (nivel, banda), el softening del arcsinh y mu/std.

    softening[l,b] = sigma_bg robusta (sigma-clipped) del canal, sobre un pool
                     de píxeles de train. Hace que el arcsinh sea adaptativo al
                     ruido: arcsinh(flux/softening) ~ lineal en el régimen de
                     ruido y comprime las partes brillantes.
    mu/std[l,b]    = media/desv. de t = arcsinh(flux/softening) en train.

    Devuelve dict con arrays de shape (5,4).
    """
    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    sub = rng.choice(n, size=min(max_stamps, n), replace=False)

    soft = np.zeros((N_LEVELS, len(BANDS)), dtype=np.float64)
    mu = np.zeros_like(soft)
    std = np.zeros_like(soft)

    for l in range(N_LEVELS):
        for b in range(len(BANDS)):
            px = X_train[sub, l, b].ravel()
            # sigma-clipping: aísla el fondo descartando píxeles de la galaxia.
            _, _, sigma_bg = sigma_clipped_stats(px, sigma=3, maxiters=5)
            sigma_bg = float(sigma_bg) if sigma_bg > 0 else 1e-3
            soft[l, b] = sigma_bg

            t = np.arcsinh(X_train[:, l, b] / sigma_bg)     # todo el train, ese canal
            mu[l, b] = t.mean()
            std[l, b] = t.std() if t.std() > 0 else 1.0

    return {"softening": soft, "mu": mu, "std": std}


def apply_norm(X, norm):
    """arcsinh(flux/softening) y luego z-score, por canal. Broadcasting sobre (N,5,4,30,30)."""
    soft = norm["softening"][None, :, :, None, None]
    mu = norm["mu"][None, :, :, None, None]
    std = norm["std"][None, :, :, None, None]
    t = np.arcsinh(X / soft)
    return ((t - mu) / std).astype(np.float32)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(n_max=None, out_name="vae_input.npz", tar_path=None, features_path=None):
    tar_path = Path(tar_path) if tar_path else TAR_PATH
    features_path = Path(features_path) if features_path else FEATURES_CSV
    print(f"Tar: {tar_path}")
    print(f"Features: {features_path}\n")
    if not tar_path.exists():
        raise FileNotFoundError(
            f"No existe el tar: {tar_path}\n"
            "Deja el archivo en data/ (ver README) o exporta COSMOS_TAR / usa --tar."
        )

    df = load_features(features_path)
    keep = set(df["desi_id"])

    print("\nCargando stamps...")
    stamps, names, n_nan = load_stamps(path=tar_path, keep_ids=keep, n_max=n_max)
    print(f"  stamps leídos (tras filtro de id): {len(names)}")
    print(f"  descartados por NaN              : {n_nan}")
    print(f"  stamps válidos                   : {len(stamps)}")

    ids, X, df_al = align(df, stamps)
    print(f"\nMuestra final alineada: N={len(ids)}, X.shape={X.shape}")

    tr, va, te = split_ids(len(ids), stratify=df_al["type"].to_numpy())
    print(f"Split  train/val/test: {len(tr)}/{len(va)}/{len(te)} "
          f"(estratificado por tipo morfológico)")

    norm = fit_norm(X[tr])
    Xn = apply_norm(X, norm)
    print(f"\nNormalizado. rango global: "
          f"[{Xn.min():.2f}, {Xn.max():.2f}], media train≈{Xn[tr].mean():.3f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / out_name
    np.savez_compressed(
        out,
        X=Xn,
        ids=np.array(ids),
        type=df_al["type"].to_numpy(),
        z=df_al["z"].to_numpy(),
        idx_train=tr, idx_val=va, idx_test=te,
        softening=norm["softening"], mu=norm["mu"], std=norm["std"],
        bands=np.array(BANDS),
    )
    # Config legible aparte.
    (OUT_DIR / "vae_input_config.json").write_text(json.dumps({
        "shape": list(Xn.shape),
        "axes": "(N, nivel[5], banda[4], 30, 30)",
        "bands": BANDS,
        "depth_cut": DEPTH_CUT, "snr_min": SNR_MIN,
        "norm": "arcsinh(flux/sigma_bg) + zscore por canal (estimado en train)",
        "split_fracs": SPLIT_FRACS, "seed": RANDOM_SEED,
    }, indent=2))
    print(f"\nGuardado: {out}")
    print(f"          {OUT_DIR / 'vae_input_config.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-max", type=int, default=None,
                    help="tope de stamps a cargar (prueba rápida)")
    ap.add_argument("--out", type=str, default="vae_input.npz")
    ap.add_argument("--tar", type=str, default=None,
                    help="ruta al tar de stamps (default: data/ o $COSMOS_TAR)")
    ap.add_argument("--features", type=str, default=None,
                    help="ruta al CSV de features (default: data/ o $COSMOS_FEATURES)")
    args = ap.parse_args()
    main(n_max=args.n_max, out_name=args.out,
         tar_path=args.tar, features_path=args.features)