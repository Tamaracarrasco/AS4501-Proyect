# Repositorio proyecto AS4501
## Representation of multiband, multiresolution images from DESI Legacy Imaging Survey.

---

## Instrucciones de configuración

### 1. Clonar el repositorio

```bash
git clone https://github.com/Tamaracarrasco/AS4501-Proyect.git
cd AS4501-Proyect
```

### 2. Crear y activar el ambiente virtual

El proyecto usa Python 3.12. Se debe crear un ambiente virtual llamado `hips-env` fuera de la carpeta del repositorio (por ejemplo, en el directorio home):

```bash
# Crear el ambiente virtual (solo la primera vez)
python3.12 -m venv ~/hips-env

# Activar el ambiente virtual (cada vez que se trabaje en el proyecto)
source ~/hips-env/bin/activate
```

> En Windows (si aplica): `~/hips-env/Scripts/activate`

### 3. Instalar las dependencias

Con el ambiente virtual activado, instalar los módulos desde el archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Data

La data debe descargarse desde la nube de Drive y dejarse de manera local. Los archivos son muy pesados para subirlos a GitHub. Colocarlos en la carpeta `/data/` para que los scripts funcionen correctamente.

---

## Pipeline del VAE (reproducible, desde terminal o VSCode)

Corre 100% local (terminal o el botón *Run* de VSCode con el intérprete `hips-env`
seleccionado). No requiere Google Colab. Usa GPU automáticamente si hay CUDA
disponible; si no, cae a CPU (lento) con un aviso.

### 0. GPU (opcional pero recomendado)

`requirements.txt` fija `torch==2.12.0+cpu`. Para usar GPU, instala la build CUDA
que corresponda a tu driver **después** de instalar los requirements, por ejemplo:

```bash
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu121
```

Verifica con: `python -c "import torch; print(torch.cuda.is_available())"`.

### 1. Datos

Deja en `/data/` el tar de stamps (`cosmos_TC_202602.tar.gz.part_aa`) y
`features_images_20260618.csv`. Si los tienes en otra carpeta, puedes apuntarlos
sin tocar el código:

```bash
export COSMOS_TAR="/ruta/a/cosmos_TC_202602.tar.gz.part_aa"
```

### 2. Preprocesamiento → `file_out_data/vae_input.npz`

Aplica los filtros (maskbits, flujo>0, profundidad, SNR, NaN), normaliza
(arcsinh + z-score por canal) y genera el tensor `(N, 5, 4, 30, 30)` con el split
train/val/test **estratificado por tipo** (semilla fija → reproducible):

```bash
python scripts/preprocess_vae.py
```

### 3. Entrenamiento del VAE

```bash
python scripts/VAE.py                 # config por defecto (z_dim=64, 200 épocas)
python scripts/VAE.py --epochs 100 --z-dim 32 --beta-final 0.5   # overrides
python scripts/VAE.py --wandb         # opcional: dashboard en vivo
```

### Salidas (en `file_out_data/`, commiteables para que el equipo las vea)

| Archivo | Contenido |
|---|---|
| `vae_metrics.csv` | una fila por época: recon, KL, beta, lr, dims activas, SSIM/PSNR,etc |
| `figures/vae/recon_*.png` | input vs reconstrucción |
| `figures/vae/prior_*.png` | samples del prior z~N(0,I) |
| `figures/vae/umap_*.png` | espacio latente (mu) coloreado por tipo y redshift |
| `vae_summary.json` | métricas finales |
| `checkpoints/vae_best.pt` | mejor modelo (menor loss de validación) |

---

## Agregar una nueva librería

Cuando se necesite instalar una nueva librería al proyecto, seguir estos pasos:

```bash
# 1. Activar el ambiente virtual
source ~/hips-env/bin/activate

# 2. Instalar la nueva librería
pip install nombre-libreria

# 3. Actualizar el archivo requirements.txt
pip freeze > requirements.txt

# 4. Subir el cambio al repositorio
git add requirements.txt
git commit -m "update requirements: add nombre-libreria"
git push
```

> Es importante actualizar `requirements.txt` antes de hacer push para que el resto del equipo pueda sincronizar el ambiente fácilmente con `pip install -r requirements.txt`.
