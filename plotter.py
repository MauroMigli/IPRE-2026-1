import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from pathlib import Path
import numpy as np
import os
import parameters

def plot_heatmaps_per_epoch_and_band(p_values_4d, channel_names, prefix="tfce", base_dir="plots"):
    """
    Generate heatmaps of empirical p-values.
    p_values_4d shape: (dest, src, band, epoch)
    """
    n_dest, n_src, n_band, n_epoch = p_values_4d.shape
    
    bands = list(parameters.F_BANDS.keys())
    
    for epoch in range(n_epoch):
        epoch_dir = os.path.join(base_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        for b_idx, band_name in enumerate(bands):
            band_pvals = p_values_4d[:, :, b_idx, epoch]
            
            # Avoid log(0)
            safe_values = np.where(band_pvals <= 0, np.finfo(float).tiny, band_pvals)
            log_values = -np.log10(safe_values) # -log10(p) is better, higher is more significant
            
            plt.figure(figsize=(10, 8))
            plt.imshow(
                log_values,
                aspect='auto',
                cmap='hot',
                vmin=0, # -log10(1) = 0
            )
            plt.colorbar(label='-log10(p-value)')
            plt.title(f"{prefix} -log10(p) - Band {band_name} - Epoch {epoch}")
            plt.xlabel('Source Channel Index')
            plt.ylabel('Destination Channel Index')
            
            filename = os.path.join(epoch_dir, f"{prefix}_heatmap_{band_name}_epoch_{epoch}.png")
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            
            # Highlight p <= 0.05
            highlight = (band_pvals <= 0.05) & np.isfinite(band_pvals)
            cmap = mcolors.ListedColormap(['black', 'red'])
            plt.figure(figsize=(10, 8))
            plt.imshow(
                highlight.astype(int),
                aspect='auto',
                cmap=cmap,
                vmin=0,
                vmax=1,
            )
            cbar = plt.colorbar(ticks=[0, 1])
            cbar.ax.set_yticklabels(['p > 0.05', 'p <= 0.05'])
            plt.title(f"Significant Edges (p <= 0.05) - Band {band_name} - Epoch {epoch}")
            plt.xlabel('Source Channel Index')
            plt.ylabel('Destination Channel Index')
            
            filename_sig = os.path.join(epoch_dir, f"{prefix}_significant_{band_name}_epoch_{epoch}.png")
            plt.savefig(filename_sig, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    p_values_dir = Path("p_values")
    if p_values_dir.exists():
        p_values = p_values_dir.iterdir()
    
        # Change scale to log for better visualization and remove vmin and vmax for absolute effect
        for p_file in p_values:
            if p_file.suffix != '.npy': continue
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
            os.makedirs("plots", exist_ok=True)
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