#!/usr/bin/env python3
"""
volume_conduction_check.py - Descomposición del efecto de grupo (FT vs PT) en
componentes de acoplamiento con rezago y de acoplamiento instantáneo (lag cero).

La conducción de volumen mezcla señales sin retardo: infla las medidas que incluyen
el término de lag cero (covarianza residual V, coherencia, coherencia parcial) y apenas
afecta las que dependen solo de los rezagos del MVAR (PDC) o que la excluyen por
construcción (coherencia imaginaria). Si la diferencia entre grupos aparece en las
medidas solo-rezago, es menos probable que se deba a diferencias de tamaño cefálico.

Medidas (promedio sobre las 56 aristas, por banda; promedio de hb y si por sujeto):
    resid_corr : |correlación| de los residuos del MVAR (SOLO lag cero)
    absCoh     : |coherencia| (incluye lag cero)
    PC         : coherencia parcial (incluye lag cero)
    dDTF       : DTF * PC (medida del pipeline)
    DTF        : DTF normalizado (depende de H, mayormente rezagos)
    PDC        : Partial Directed Coherence (solo coeficientes A, sin V)
    iCoh       : |parte imaginaria de la coherencia| (robusta a conducción de volumen)
Contraste: Mann-Whitney FT vs PT y cociente de medias PT/FT.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import parameters
from src.preprocessing import get_valid_subjects, load_roi_epochs
from src.connectivity import fit_mvar_multitrial

MEASURES = {
    'resid_corr': 'solo lag cero',
    'absCoh': 'incluye lag cero',
    'PC': 'incluye lag cero',
    'dDTF': 'incluye lag cero (PC)',
    'DTF': 'mayormente rezagos',
    'PDC': 'solo rezagos',
    'iCoh': 'excluye lag cero',
}


def spectral_measures(A, V, sampling_freq, freqs):
    """Todas las medidas espectrales de un MVAR, cada una (n_fs, dest, src)."""
    p, n_rois, _ = A.shape
    exp_matrix = np.exp(-2j * np.pi * np.outer(freqs, np.arange(1, p + 1)) / sampling_freq)
    A_f = np.eye(n_rois) - np.einsum('fp,pij->fij', exp_matrix, A)
    H = np.linalg.inv(A_f)
    S = H @ V @ np.conj(np.transpose(H, (0, 2, 1)))

    H2 = np.abs(H) ** 2
    DTF = np.sqrt(H2 / H2.sum(axis=2, keepdims=True))

    G = np.linalg.inv(S)
    G_diag = np.abs(np.diagonal(G, axis1=1, axis2=2))
    PC = np.abs(G) / np.sqrt(G_diag[:, :, None] * G_diag[:, None, :])

    A2 = np.abs(A_f) ** 2
    PDC = np.sqrt(A2 / A2.sum(axis=1, keepdims=True))  # normalizado por columna (origen)

    S_diag = np.real(np.diagonal(S, axis1=1, axis2=2))
    coh = S / np.sqrt(S_diag[:, :, None] * S_diag[:, None, :])

    return {'DTF': DTF, 'PC': PC, 'dDTF': DTF * PC, 'PDC': PDC, 'iCoh': np.abs(coh.imag), 'absCoh': np.abs(coh)}


def recording_summary(filepath, p, roi_method):
    """Medidas por banda (promedio sobre aristas) de un registro: {medida: (n_bands,)}."""
    fit = fit_mvar_multitrial(load_roi_epochs(filepath, method=roi_method), p)
    off_diag = ~np.eye(fit['V'].shape[0], dtype=bool)
    freqs = parameters.FS_GLOBAL

    std = np.sqrt(np.diag(fit['V']))
    resid_corr = np.abs((fit['V'] / np.outer(std, std))[off_diag]).mean()
    out = {'resid_corr': np.full(len(parameters.F_BANDS), resid_corr)}

    for name, values in spectral_measures(fit['A'], fit['V'], parameters.FS_TARGET, freqs).items():
        out[name] = np.array([values[(freqs >= lo) & (freqs <= hi)][:, off_diag].mean()
                              for lo, hi in parameters.F_BANDS.values()])
    return out


def main():
    parser = argparse.ArgumentParser(description="Chequeo de conducción de volumen: efecto de grupo con rezago vs lag cero")
    parser.add_argument("--p", type=int, default=parameters.P_OPTIMO)
    parser.add_argument("--roi-method", type=str, choices=['mean', 'pca'], default=parameters.ROI_EXTRACTION_METHOD)
    parser.add_argument("--output-dir", type=str, default="plots")
    args = parser.parse_args()

    valid_subjects = get_valid_subjects()
    values = {name: [] for name in MEASURES}
    groups = []
    for kid_id, info in valid_subjects.items():
        hb = recording_summary(info['hb'], args.p, args.roi_method)
        si = recording_summary(info['si'], args.p, args.roi_method)
        for name in MEASURES:
            values[name].append((hb[name] + si[name]) / 2)
        groups.append(info['group'])
    groups = np.array(groups)
    values = {name: np.array(v) for name, v in values.items()}  # (n_subj, n_bands)

    band_names = list(parameters.F_BANDS)
    ratios = np.zeros((len(MEASURES), len(band_names)))
    pvals = np.zeros_like(ratios)

    print("\n==========================================================================")
    print(f" CONDUCCIÓN DE VOLUMEN: EFECTO DE GRUPO POR TIPO DE MEDIDA [ROI: {args.roi_method.upper()}]")
    print(" Cociente PT/FT de las medias y p de Mann-Whitney")
    print("==========================================================================")
    header = " | ".join(f"{b}: PT/FT     p" for b in band_names)
    print(f"  {'medida':11s} {'lag cero':22s} | {header}")
    for m_idx, (name, zero_lag) in enumerate(MEASURES.items()):
        cells = []
        for b_idx in range(len(band_names)):
            ft = values[name][groups == 'FT', b_idx]
            pt = values[name][groups == 'PT', b_idx]
            ratios[m_idx, b_idx] = pt.mean() / ft.mean()
            pvals[m_idx, b_idx] = mannwhitneyu(ft, pt).pvalue
            cells.append(f"{ratios[m_idx, b_idx]:9.2f} {pvals[m_idx, b_idx]:9.4f}")
        print(f"  {name:11s} {zero_lag:22s} | " + " | ".join(cells))

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"volume_conduction_check_{args.roi_method}.npz")
    np.savez(out_file, measures=np.array(list(MEASURES)), band_names=np.array(band_names), ratio_pt_ft=ratios, p_mannwhitney=pvals)
    print(f"\n[INFO] Resumen por grupo guardado en: {out_file}")


if __name__ == "__main__":
    main()
