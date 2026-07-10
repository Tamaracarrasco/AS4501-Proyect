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

Aplica los filtros (maskbits, flujo>0, profundidad, SNR, NaN), normaliza con
"otro" (signed sqrt scaling: `x/1000 -> sign(x)*(sqrt(sign(x)*x + 1) - 1)`, por
canal banda×nivel) y genera el tensor `(N, 5, 4, 30, 30)` con el split
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

## Alternativa: entrenar el VAE en Colab Pro desde VSCode

Si no tienes GPU local (o quieres una más rápida), puedes correr `scripts/VAE.py`
contra un runtime de **Google Colab Pro** sin salir de VSCode, usando la extensión
oficial **"Colab"** (publisher: Google) del Marketplace. A diferencia de abrir
colab.research.google.com en el navegador, esta extensión conecta un notebook
`.ipynb` abierto en VSCode directamente a un runtime remoto de Colab (con GPU),
manteniendo el editor, el resaltado y el resto de tus extensiones.

1. **Instalar la extensión**: en VSCode, Extensions (`Ctrl+Shift+X`) → buscar
   "Colab" (publisher Google) → Install.
2. **Iniciar sesión** con la cuenta de Google que tiene la suscripción Colab Pro.
3. **Abrir un notebook**: crea o abre un `.ipynb` (p.ej. `scripts/run_vae_colab.ipynb`).
   En el selector de kernel (arriba a la derecha) elige el runtime de **Colab**
   en vez de un intérprete local.
4. **Elegir GPU Pro**: dentro de la sesión de Colab conectada, `Runtime > Change
   runtime type` para elegir GPU (con Pro tienes acceso a GPUs mejores y más RAM
   que en el tier gratuito).
5. **Traer el repo y los datos** (el runtime de Colab arranca vacío) — en la
   primera celda:
   ```python
   !git clone https://github.com/Tamaracarrasco/AS4501-Proyect.git
   %cd AS4501-Proyect
   !pip install -r requirements.txt
   ```
   Los datos de `/data/` pesan varios GB y no están en GitHub: súbelos a tu
   Google Drive una vez y móntalo en cada sesión:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
   luego usa `--tar` / `--features` (o `$COSMOS_TAR` / `$COSMOS_FEATURES`) para
   apuntar a la ruta dentro de `/content/drive/MyDrive/...`.
6. **Verificar la GPU**:
   ```python
   !nvidia-smi
   import torch; print(torch.cuda.is_available())
   ```
   `VAE.py` detecta CUDA automáticamente (no hay que tocar el código).
7. **Preprocesar y entrenar**, igual que en local pero como celdas de shell
   (`!comando`) — corren en la máquina remota de Colab, no en tu laptop:
   ```python
   !python scripts/preprocess_vae.py
   !python scripts/VAE.py --epochs 200 --z-dim 64
   ```
8. **Persistir resultados**: el runtime de Colab es efímero (se borra al
   desconectar). Apunta `--out-dir` a tu Drive montado para no perder
   checkpoints/figuras entre sesiones:
   ```python
   !python scripts/VAE.py --out-dir /content/drive/MyDrive/AS4501-Proyect/file_out_data
   ```
9. Al terminar, descarga (o copia desde Drive) el `file_out_data/` resultante a
   tu clon local y comitea lo que quieras compartir con el equipo.

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
