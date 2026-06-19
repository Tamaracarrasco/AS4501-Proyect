# ahora vamos a cargar las imágenes.
import tarfile, io
import numpy as np
import matplotlib.pyplot as plt

# Cambiar estos directorios.

DDF = "COSMOS"
IMAGES_PATH = f"../data/"
IMAGES_FILE = f"cosmos_TC_202602.tar.gz.part_aa"

def load_all(path):
    """Read all .npy files from a tar.gz archive, tolerates truncated files."""
    arrays, names = [], []
    try:
        with tarfile.open(path, "r|gz") as tar:  # r|gz = streaming, handles truncation
            for m in tar:
                if not m.name.endswith(".npy"):
                    continue
                f = tar.extractfile(m)
                if f is None:
                    continue
                arrays.append(np.load(io.BytesIO(f.read())))
                names.append(m.name.split("/")[-1].replace(".npy", ""))
    except (EOFError, tarfile.ReadError):
        print(f"Archivo truncado — se recuperaron {len(arrays)} galaxias antes del corte.")
    return arrays, names

#%%
arrays, names = load_all(IMAGES_PATH + IMAGES_FILE)
print(names)

# actualemte son 32mil imagenes pero las fetures solo tienen 29mil  registros nms