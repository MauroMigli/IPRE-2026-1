import numpy as np
import os
import json
from pathlib import Path
try:
    from statsmodels.tsa.api import VAR
except ImportError:
    VAR = None

try:
    import mne
except ImportError:
    mne = None

import parameters
from src.preprocessing import clean_epochs
from src.connectivity import aggregate_channels_to_rois

def evaluate_epoch_orders(roi_epoch_data, max_p=15):
    """
    Evalúa los criterios de información (AIC, BIC, FPE, HQIC) para una época dada
    a través de los rezagos p = 1, ..., max_p.
    
    Parámetros:
    -----------
    roi_epoch_data: np.ndarray de forma (n_times, n_rois)
    max_p: rezago máximo a evaluar
    
    Retorna:
    --------
    dict con arrays de criterios para cada lag p
    """
    if VAR is None:
        raise ImportError(
            "statsmodels no está disponible en este entorno de Python.\n"
            "Por favor, activa el entorno virtual (ej: source ../.venv/bin/activate) "
            "o instala statsmodels con: pip install statsmodels"
        )

    n_times, n_rois = roi_epoch_data.shape
    safe_max_p = min(max_p, (n_times - 1) // n_rois)
    lags = np.arange(1, safe_max_p + 1)
    
    aic_list = []
    bic_list = []
    fpe_list = []
    hqic_list = []
    
    model = VAR(roi_epoch_data)
    for p in lags:
        try:
            fitted = model.fit(maxlags=p, trend='c')
            aic_list.append(fitted.aic)
            bic_list.append(fitted.bic)
            fpe_list.append(fitted.fpe)
            hqic_list.append(fitted.hqic)
        except Exception:
            aic_list.append(np.nan)
            bic_list.append(np.nan)
            fpe_list.append(np.nan)
            hqic_list.append(np.nan)
            
    return {
        'lags': lags,
        'aic': np.array(aic_list),
        'bic': np.array(bic_list),
        'fpe': np.array(fpe_list),
        'hqic': np.array(hqic_list)
    }

def evaluate_subject_orders(filepath, rois_dict=None, max_p=15, method='mean'):
    """
    Calcula las curvas de orden MVAR para todas las épocas de un sujeto específico.
    """
    if rois_dict is None:
        rois_dict = parameters.ROIS
        
    epochs = clean_epochs(mne.io.read_epochs_eeglab(filepath, verbose=False))
    ch_names = epochs.ch_names
    data = epochs.get_data(copy=False)  # (n_epochs, n_channels, n_times)
    
    roi_data = aggregate_channels_to_rois(data, ch_names, rois_dict, method=method)
    n_epochs = roi_data.shape[0]
    
    all_aic = []
    all_bic = []
    lags = None
    
    for ep in range(n_epochs):
        res = evaluate_epoch_orders(roi_data[ep].T, max_p=max_p)
        lags = res['lags']
        all_aic.append(res['aic'])
        all_bic.append(res['bic'])
        
    return {
        'lags': lags,
        'aic': np.array(all_aic),  # (n_epochs, len(lags))
        'bic': np.array(all_bic)   # (n_epochs, len(lags))
    }

def find_dataset_optimal_order(valid_subjects, rois_dict=None, max_p=15, method='mean', output_dir="plots"):
    """
    Itera sobre todos los sujetos y condiciones válidas del dataset, evalúa AIC y BIC
    en cada época multivariada sobre los super-nodos y reporta el orden óptimo global.
    
    Guarda los resultados estructurados en plots/mvar_order_selection.json.
    """
    if rois_dict is None:
        rois_dict = parameters.ROIS
        
    os.makedirs(output_dir, exist_ok=True)
    
    total_aic = []
    total_bic = []
    common_lags = None
    
    print(f"\n--- Evaluando Orden MVAR Óptimo (AIC/BIC) sobre {len(valid_subjects)} sujetos ---")
    for kid_id, info in valid_subjects.items():
        for cond in ['hb', 'si']:
            fpath = info[cond]
            if fpath and os.path.exists(fpath):
                print(f"  -> Evaluando sujeto {kid_id} (condición {cond.upper()})...", flush=True)
                res = evaluate_subject_orders(fpath, rois_dict=rois_dict, max_p=max_p, method=method)
                if common_lags is None:
                    common_lags = res['lags']
                total_aic.append(res['aic'])
                total_bic.append(res['bic'])
                
    if not total_aic:
        print("  [AVISO] No se encontraron archivos de sujetos para evaluar.")
        return None
        
    total_aic = np.vstack(total_aic)  # (total_epochs, len(lags))
    total_bic = np.vstack(total_bic)
    
    mean_aic = np.nanmean(total_aic, axis=0)
    sem_aic = np.nanstd(total_aic, axis=0) / np.sqrt(np.sum(~np.isnan(total_aic), axis=0))
    
    mean_bic = np.nanmean(total_bic, axis=0)
    sem_bic = np.nanstd(total_bic, axis=0) / np.sqrt(np.sum(~np.isnan(total_bic), axis=0))
    
    aic_best_per_epoch = common_lags[np.nanargmin(total_aic, axis=1)]
    bic_best_per_epoch = common_lags[np.nanargmin(total_bic, axis=1)]
    
    opt_p_aic = int(common_lags[np.nanargmin(mean_aic)])
    opt_p_bic = int(common_lags[np.nanargmin(mean_bic)])
    
    results = {
        'lags': common_lags.tolist(),
        'mean_aic': mean_aic.tolist(),
        'sem_aic': sem_aic.tolist(),
        'mean_bic': mean_bic.tolist(),
        'sem_bic': sem_bic.tolist(),
        'aic_votes': aic_best_per_epoch.tolist(),
        'bic_votes': bic_best_per_epoch.tolist(),
        'optimal_p_aic': opt_p_aic,
        'optimal_p_bic': opt_p_bic,
        'recommended_p': opt_p_bic,
        'total_epochs_evaluated': int(total_aic.shape[0])
    }
    
    out_file = os.path.join(output_dir, "mvar_order_selection.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print("\n==========================================================================")
    print(" RESULTADOS DE SELECCIÓN DE ORDEN MVAR:")
    print(f"  * Total de épocas analizadas: {results['total_epochs_evaluated']}")
    print(f"  * Orden óptimo según AIC: p = {opt_p_aic}")
    print(f"  * Orden óptimo según BIC: p = {opt_p_bic} (Recomendado para evitar overfitting)")
    print(f"  * Métricas completas guardadas en: {out_file}")
    print("==========================================================================")
    
    return results
