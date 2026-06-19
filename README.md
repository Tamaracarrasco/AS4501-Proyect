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
