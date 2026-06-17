from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from pathlib import Path
import numpy as np

p_values = Path("p_values").iterdir()

# Change scale to log for better visualization and remove vmin and vmax for absolute effect
for p_file in p_values:
    p_values_array = np.load(p_file)
    # Avoid non-finite/zero values before taking log10.
    finite_mask = np.isfinite(p_values_array)
    if not np.any(finite_mask):
        continue
    safe_values = np.where(
        finite_mask,
        np.maximum(p_values_array, np.finfo(float).tiny),
        np.nan,
    )
    log_values = np.log10(safe_values)
    vmin = np.nanmin(log_values)
    vmax = np.nanmax(log_values)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        log_values,
        aspect='auto',
        cmap='viridis',
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(label='log10(p-value)')
    plt.title(f"P-values for {p_file.stem}")
    plt.xlabel('Channel Index')
    plt.ylabel('Channel Index')
    plt.savefig(f'./plots/{p_file.stem}.png')
    plt.close()

    threshold = 0.001 / 2116
    highlight = (p_values_array <= threshold) & np.isfinite(p_values_array)
    cmap = mcolors.ListedColormap(['black', 'red'])
    plt.figure(figsize=(10, 6))
    plt.imshow(
        highlight.astype(int),
        aspect='auto',
        cmap=cmap,
        vmin=0,
        vmax=1,
    )
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels([f'> {threshold}', f'<= {threshold}'])
    plt.title(f"P-values <= {threshold} for {p_file.stem}")
    plt.xlabel('Channel Index')
    plt.ylabel('Channel Index')
    plt.savefig(f'./plots/{p_file.stem}_thresh_{threshold}.png')
    plt.close()