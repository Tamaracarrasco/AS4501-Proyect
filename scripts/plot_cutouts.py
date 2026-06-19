"""
Plot multiresolution cutouts for a single galaxy.
Array shape per galaxy: (B, 5, 30, 30) — bands × resolution levels × H × W.
Plot layout: rows = resolution levels (0→4), columns = bands (g/r/i/z).
"""

import tarfile, io
import numpy as np
import matplotlib.pyplot as plt

ARCHIVE = "/mnt/d/thesis_files/files/galaxies_samples/COSMOS/images/cosmos_TC_202602.tar.gz.part_aa"
BAND_LABELS = ["g", "r", "i", "z"]
N_LEVELS = 5


def load_n(path, n):
    """Read the first n .npy files from a split/truncated tar.gz archive."""
    arrays, names = [], []
    try:
        with tarfile.open(path, "r|gz") as tar:
            for m in tar:
                if not m.name.endswith(".npy"):
                    continue
                f = tar.extractfile(m)
                if f is None:
                    continue
                arrays.append(np.load(io.BytesIO(f.read())))
                names.append(m.name.split("/")[-1].replace(".npy", ""))
                if len(arrays) == n:
                    break
    except (EOFError, tarfile.ReadError):
        pass
    return arrays, names


def normalise(img):
    vmin, vmax = np.percentile(img, [1, 99])
    return np.clip((img - vmin) / (vmax - vmin + 1e-10), 0, 1)


def plot_galaxy(arr, name, band_labels=BAND_LABELS, n_levels=N_LEVELS, save_path=None):
    """
    Plot one galaxy: rows = resolution levels, columns = bands.
    arr shape: (B, 5, 30, 30)
    """
    n_bands = arr.shape[0]
    labels = band_labels[:n_bands]

    fig, axes = plt.subplots(n_bands, n_levels, figsize=(n_levels * 2.5, n_bands * 2.5))
    fig.suptitle(f"Galaxy {name} — {n_bands} bands × {n_levels} resolution levels", fontsize=12)

    for row in range(n_bands):
        for col in range(n_levels):
            ax = axes[row, col]
            ax.imshow(normalise(arr[row, col]), origin="lower", cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"level {col}", fontsize=10)
        axes[row, 0].set_ylabel(labels[row], fontsize=10, rotation=0, labelpad=30, va="center")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"Saved: {save_path}")
    plt.show()


arrays, names = load_n(ARCHIVE, 1)
plot_galaxy(arrays[0], names[0],
            save_path="/home/tamara/thesis-proyect/bricks_ddf_lsst/scripts/cosmos_cutouts.png")
