"""
En este script se vera si la muestra de los stamps
es representativa o no.
"""
#%%
# Importación de librerías importantes

import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt

import astropy 
from pathlib import Path 
from astropy.table import Table 
from astropy.coordinates import SkyCoord 
import astropy.units as u

import os, urllib.request, urllib.error, time, http.client
from astropy.table import unique, vstack

import tarfile, io
import numpy as np
import matplotlib.pyplot as plt
import re

#%%

FILE_PATH_COSMOS = "../../data/COSMOS_galaxies_20260618.txt"
df_cosmos = pd.read_csv(FILE_PATH_COSMOS, sep=",")
df_cosmos.info()

#%%
df_cosmos.columns.tolist()

# se renombra la columna id por desi_id
df_cosmos.rename(columns={"targetid": "desi_id"}, inplace=True)

# se eliminan columnas inutiles
df_cosmos.drop(columns=["zwarn", "spectype", "survey", "release", "dist_deg", "Separation"], inplace=True)

#%%
FILE_PATH_FEATURES = "../../data/features_images_20260618.csv"
df_features_images = pd.read_csv(FILE_PATH_FEATURES, sep=",")
#%%
#%%
print(f"Tamaño df_cosmos: {df_cosmos.shape}")
print(f"Tamaño df_feat: {df_features_images.shape}")

#%%
#### COMPARACIÓN DISTRIBUCIONES DE MUESTRAS

def compare_dist(col, bins=80, log_x=False, xlabel=None):
    """
    Histogramas normalizados de `col` en df_cosmos y df_features_images.
    log_x=True: aplica log10 filtrando valores <= 0 antes de graficar.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    label_x = xlabel or (f"log10({col})" if log_x else col)

    for ax, df, title in zip(
        axes,
        [df_cosmos, df_features_images],
        [f"df_cosmos  (n={len(df_cosmos):,})", f"df_features_images  (n={len(df_features_images):,})"],
    ):
        vals = df[col].dropna()
        if log_x:
            vals = np.log10(vals[vals > 0])
        ax.hist(vals, bins=bins, density=True, edgecolor="k", linewidth=0.2)
        ax.set_title(title)
        ax.set_xlabel(label_x)
        ax.set_ylabel("densidad")

    plt.suptitle(col)
    plt.tight_layout()
    plt.show()

#%%
# Redshift espectroscópico
compare_dist("z", bins=80)

#%%
# Flujos por banda (log scale)
flux_cols = [
    c for c in df_cosmos.columns
    if re.fullmatch(r"flux_\w{1,2}", c) and c in df_features_images.columns
]
print(f"Columnas de flujo: {flux_cols}")

for col in flux_cols:
    compare_dist(col, bins=80, log_x=True)

#%%
# Fracción de flujos negativos o nulos por banda
neg_frac = pd.DataFrame({
    "df_cosmos":          [(df_cosmos[c] <= 0).mean() for c in flux_cols],
    "df_features_images": [(df_features_images[c] <= 0).mean() for c in flux_cols],
}, index=flux_cols).round(4)
print(neg_frac)

#%%
# Magnitudes AB (solo flujo > 0)
def flux_to_mag(flux_series):
    pos = flux_series.copy().astype(float)
    pos[pos <= 0] = np.nan
    return 22.5 - 2.5 * np.log10(pos)

for col in flux_cols:
    mag_col = col.replace("flux_", "mag_")
    df_cosmos[mag_col] = flux_to_mag(df_cosmos[col])
    df_features_images[mag_col] = flux_to_mag(df_features_images[col])
    compare_dist(mag_col, bins=80, xlabel="magnitud AB")