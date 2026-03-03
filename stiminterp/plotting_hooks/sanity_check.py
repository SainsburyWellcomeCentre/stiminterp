import matplotlib.pyplot as plt
import numpy as np


def create_sanitycheck_axes(nchannel: int):
    """
    Create axes for stim visualisation.

    Layout:
        rows = nchannel
        cols = 3  (Uncorrected | Bad mask | Corrected)

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of shape (nchannel, 3)
    """

    ncols = 3

    fig, axes = plt.subplots(
        nchannel,
        ncols,
        figsize=(5 * ncols, 4 * nchannel),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    # Ensure 2D array even if nchannel == 1
    if nchannel == 1:
        axes = axes[None, :]

    # Column titles (only top row)
    axes[0, 0].set_title("Uncorrected")
    axes[0, 1].set_title("NaN mask")
    axes[0, 2].set_title("Corrected")

    return fig, axes


def plot_removal(
    axes_row,
    frame: int,
    y_frac_start: float,
    y_frac_stop: float,
    uncorrected: np.ndarray,
    corrected: np.ndarray,
    bad_mask: np.ndarray,
    channel: int,  # <-- NEW
    cmap: str = "gray",
):
    raw = uncorrected[frame]
    corr = corrected[frame]
    mask = bad_mask[frame]

    Y = uncorrected.shape[1]
    y_min = int(y_frac_start * Y)
    y_max = int(y_frac_stop * Y)

    vmin = np.percentile(raw, 1)
    vmax = np.percentile(raw, 99.5)

    # --- Uncorrected ---
    axes_row[0].set_ylabel(f"Ch {channel}\nFrame {frame}")
    axes_row[0].imshow(raw, cmap=cmap, vmin=vmin, vmax=vmax)
    axes_row[0].axhline(y_min, c="r", lw=2)
    axes_row[0].axhline(y_max, c="r", lw=2)
    axes_row[0].axis("off")

    # --- Bad mask ---
    axes_row[1].imshow(mask, cmap="Reds")
    axes_row[1].axhline(y_min, c="k", lw=2)
    axes_row[1].axhline(y_max, c="k", lw=2)
    axes_row[1].axis("off")

    # --- Corrected ---
    axes_row[2].imshow(corr, cmap=cmap, vmin=vmin, vmax=vmax)
    # axes_row[2].axhline(y_min, c="r", lw=2)
    # axes_row[2].axhline(y_max, c="r", lw=2)
    axes_row[2].axis("off")
