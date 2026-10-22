import numpy as np
import os
import json

import parameters
from src.preprocessing import load_roi_epochs
from src.connectivity import fit_mvar_multitrial


def evaluate_orders(roi_epochs, max_p=15):
    """
    Evalúa AIC y BIC de un MVAR multi-ensayo para p = 1, ..., max_p.
    Todos los órdenes se estiman sobre la MISMA muestra (respuestas desde t = max_p
    en cada época), condición necesaria para que los criterios sean comparables.

    Parámetros:
    -----------
    roi_epochs: np.ndarray (n_epochs, n_rois, n_times)
    max_p: rezago máximo a evaluar

    Retorna:
    --------
    dict con lags, aic, bic y max_root para cada p
    """
    n_epochs, n_rois, n_times = roi_epochs.shape
    lags = np.arange(1, max_p + 1)
    aic, bic, roots = [], [], []

    for p in lags:
        fit = fit_mvar_multitrial(roi_epochs, p, p_start=max_p)
        n_obs = fit['n_obs']
        n_params = n_rois * (n_rois * p + 1)
        _, logdet = np.linalg.slogdet(fit['sigma_ml'])
        aic.append(logdet + 2.0 * n_params / n_obs)
        bic.append(logdet + np.log(n_obs) * n_params / n_obs)
        roots.append(fit['max_root'])

    return {'lags': lags, 'aic': np.array(aic), 'bic': np.array(bic), 'max_root': np.array(roots)}


def find_dataset_optimal_order(valid_subjects, rois_dict=None, max_p=15, method='mean', output_dir="plots"):
    """
    Evalúa AIC/BIC sobre cada registro (sujeto x condición) y reporta el orden óptimo global.
    Cada registro aporta un "voto" (su p óptimo) y una curva; el orden recomendado
    es el mínimo de la curva BIC promedio.

    Guarda los resultados en plots/mvar_order_selection_<method>.json.
    """
    if rois_dict is None:
        rois_dict = parameters.ROIS

    os.makedirs(output_dir, exist_ok=True)

    total_aic, total_bic, total_roots = [], [], []
    lags = None

    print(f"\n--- Evaluando Orden MVAR Óptimo (AIC/BIC, {parameters.FS_TARGET:.0f} Hz) sobre {len(valid_subjects)} sujetos ---")
    for kid_id, info in valid_subjects.items():
        for cond in ['hb', 'si']:
            fpath = info[cond]
            if fpath and os.path.exists(fpath):
                print(f"  -> Evaluando sujeto {kid_id} (condición {cond.upper()})...", flush=True)
                roi_data = load_roi_epochs(fpath, rois_dict, method=method)
                res = evaluate_orders(roi_data, max_p=max_p)
                lags = res['lags']
                total_aic.append(res['aic'])
                total_bic.append(res['bic'])
                total_roots.append(res['max_root'])

    if not total_aic:
        print("  [AVISO] No se encontraron archivos de sujetos para evaluar.")
        return None

    total_aic = np.vstack(total_aic)  # (n_registros, len(lags))
    total_bic = np.vstack(total_bic)
    total_roots = np.vstack(total_roots)

    mean_aic = total_aic.mean(axis=0)
    mean_bic = total_bic.mean(axis=0)
    opt_p_aic = int(lags[np.argmin(mean_aic)])
    opt_p_bic = int(lags[np.argmin(mean_bic)])

    results = {
        'method': method,
        'fs': parameters.FS_TARGET,
        'lags': lags.tolist(),
        'mean_aic': mean_aic.tolist(),
        'sem_aic': (total_aic.std(axis=0) / np.sqrt(len(total_aic))).tolist(),
        'mean_bic': mean_bic.tolist(),
        'sem_bic': (total_bic.std(axis=0) / np.sqrt(len(total_bic))).tolist(),
        'aic_votes': lags[np.argmin(total_aic, axis=1)].tolist(),
        'bic_votes': lags[np.argmin(total_bic, axis=1)].tolist(),
        'stable_fraction': (total_roots < 1).mean(axis=0).tolist(),
        'optimal_p_aic': opt_p_aic,
        'optimal_p_bic': opt_p_bic,
        'recommended_p': opt_p_bic,
        'total_recordings_evaluated': int(total_aic.shape[0])
    }

    out_file = os.path.join(output_dir, f"mvar_order_selection_{method}.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)

    print("\n==========================================================================")
    print(" RESULTADOS DE SELECCIÓN DE ORDEN MVAR:")
    print(f"  * Registros analizados: {results['total_recordings_evaluated']}")
    print(f"  * Orden óptimo según AIC: p = {opt_p_aic}")
    print(f"  * Orden óptimo según BIC: p = {opt_p_bic} (Recomendado para evitar overfitting)")
    print(f"  * Fracción de modelos estables en p = {opt_p_bic}: {results['stable_fraction'][opt_p_bic - 1]:.2f}")
    print(f"  * Métricas completas guardadas en: {out_file}")
    print("==========================================================================")

    return results
