#!/usr/bin/env python3
"""
power_contrast.py - Chequeo de cordura a nivel de potencia espectral por ROI.

Antes de interpretar un resultado nulo de conectividad conviene saber si el estímulo
(heartbeat vs silencio) produce ALGÚN efecto en el EEG. Este script usa exactamente el
mismo preprocesamiento que el pipeline (ROIs -> des-tendencia -> remuestreo) y contrasta
la log-potencia por banda y ROI:
  1. Efecto de condición: t pareado de (hb - si) sobre todos los sujetos.
  2. Interacción: Welch FT vs PT sobre (hb - si).
FDR sobre las n_bandas x n_ROIs pruebas de cada contraste.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
from scipy.signal import welch
from scipy.stats import ttest_1samp, ttest_ind

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import parameters
from src.preprocessing import get_valid_subjects, load_roi_epochs
from src.statistics import fdrcorrect_bh


def band_log_power(roi_epochs):
    """(n_epochs, n_rois, n_times) -> (n_bands, n_rois) log10 de la potencia media por banda."""
    f, P = welch(roi_epochs, fs=parameters.FS_TARGET, nperseg=roi_epochs.shape[-1], axis=-1)
    P = P.mean(axis=0)  # promedio sobre épocas: (n_rois, n_f)
    return np.array([np.log10(P[:, (f >= lo) & (f <= hi)].mean(axis=1)) for lo, hi in parameters.F_BANDS.values()])


def main():
    parser = argparse.ArgumentParser(description="Contraste de potencia espectral por ROI (chequeo de cordura)")
    parser.add_argument("--roi-method", type=str, choices=['mean', 'pca'], default=parameters.ROI_EXTRACTION_METHOD)
    parser.add_argument("--output-dir", type=str, default="plots")
    args = parser.parse_args()

    valid_subjects = get_valid_subjects()
    D, groups = [], []
    for kid_id, info in valid_subjects.items():
        hb = band_log_power(load_roi_epochs(info['hb'], method=args.roi_method))
        si = band_log_power(load_roi_epochs(info['si'], method=args.roi_method))
        D.append(hb - si)
        groups.append(info['group'])
    D = np.array(D)  # (n_subj, n_bands, n_rois)
    groups = np.array(groups)

    t_cond, p_cond = ttest_1samp(D, 0.0, axis=0)
    q_cond = fdrcorrect_bh(p_cond.ravel()).reshape(p_cond.shape)
    t_grp, p_grp = ttest_ind(D[groups == 'FT'], D[groups == 'PT'], axis=0, equal_var=False)
    q_grp = fdrcorrect_bh(p_grp.ravel()).reshape(p_grp.shape)

    print("\n==========================================================================")
    print(f" POTENCIA ESPECTRAL (log10) POR ROI [ROI: {args.roi_method.upper()}] | FT = {np.sum(groups == 'FT')}, PT = {np.sum(groups == 'PT')}")
    print(" t > 0 en condición: hb > si | t > 0 en grupo: efecto (hb - si) mayor en FT que en PT")
    print("==========================================================================")
    for b_idx, b_name in enumerate(parameters.F_BANDS):
        for r_idx, roi in enumerate(parameters.ROI_NAMES):
            print(f"  {b_name:5s} {roi:18s} condición t={t_cond[b_idx, r_idx]:+5.2f} q={q_cond[b_idx, r_idx]:.3f} | "
                  f"FT vs PT t={t_grp[b_idx, r_idx]:+5.2f} p={p_grp[b_idx, r_idx]:.3f} q={q_grp[b_idx, r_idx]:.3f}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"power_contrast_{args.roi_method}.npz")
    np.savez(out_file, D=D, groups=groups, band_names=np.array(list(parameters.F_BANDS)),
             node_names=np.array(parameters.ROI_NAMES), t_cond=t_cond, p_cond=p_cond, q_cond=q_cond,
             t_group=t_grp, p_group=p_grp, q_group=q_grp)
    print(f"\n[INFO] Resultados guardados en: {out_file}")


if __name__ == "__main__":
    main()
