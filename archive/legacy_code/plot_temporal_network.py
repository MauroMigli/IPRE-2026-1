#!/usr/bin/env python3
"""
plot_temporal_network.py - Generador de redes 3D temporales interactivas (Plotly HTML)
con Slider y botón de reproducción para visualizar la evolución de la conectividad en EEG.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np

# Asegurar que la raíz del proyecto esté en sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import parameters
from src.statistics import get_roi_3d_centroids
from src.visualization import export_interactive_temporal_3d_network

def main():
    parser = argparse.ArgumentParser(
        description="Genera grafos 3D interactivos con línea de tiempo/slider para una o varias bandas de frecuencia."
    )
    parser.add_argument(
        "--band",
        type=str,
        default="both",
        choices=["Delta", "Theta", "Alpha", "Beta", "Gamma", "both", "all"],
        help="Banda de frecuencia a graficar ('both' genera Delta y Gamma)"
    )
    parser.add_argument(
        "--p-threshold",
        type=float,
        default=0.05,
        help="Umbral de significancia estadística (default: 0.05)"
    )
    parser.add_argument(
        "--roi-method",
        type=str,
        choices=["mean", "pca"],
        default="mean",
        help="Método intra-ROI: 'mean' o 'pca' (default: 'mean')"
    )
    parser.add_argument(
        "--p-file",
        type=str,
        default=None,
        help="Ruta al archivo .npy con los p-valores calculados (None = automático según --roi-method)"
    )
    parser.add_argument(
        "--nodes-file",
        type=str,
        default="plots/p_values_rois_node_names.npy",
        help="Ruta al archivo .npy con los nombres de los super-nodos (default: plots/p_values_rois_node_names.npy)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directorio de destino para los archivos .html (default: plots)"
    )
    
    args = parser.parse_args()
    
    # Resolver ruta de p-file si es None
    if args.p_file is None:
        candidate = f"plots/p_values_rois_{args.roi_method}_naive.npy"
        if os.path.exists(candidate):
            args.p_file = candidate
        else:
            args.p_file = "plots/p_values_rois_naive.npy"
            
    if not os.path.exists(args.p_file):
        print(f"[ERROR] No se encontró el archivo de p-valores: {args.p_file}")
        print("Ejecuta primero el pipeline: python run_pipeline.py --method all --use-rois")
        sys.exit(1)
        
    p_values = np.load(args.p_file)  # (dest, src, band, epoch)
    
    if os.path.exists(args.nodes_file):
        node_names = np.load(args.nodes_file, allow_pickle=True)
    else:
        node_names = parameters.ROI_NAMES
        
    band_names = list(parameters.F_BANDS.keys())
    coords_3d, _ = get_roi_3d_centroids(parameters.ELP_FILE, parameters.ROIS)
    
    # Determinar qué bandas procesar
    if args.band == "both":
        target_bands = ["Delta", "Gamma"]
    elif args.band == "all":
        target_bands = band_names
    else:
        target_bands = [args.band]
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n==========================================================================")
    print(" GENERADOR DE REDES 3D TEMPORALES INTERACTIVAS")
    print(f" Nodos: {len(node_names)} super-nodos | Épocas: {p_values.shape[3]}")
    print(f" Umbral de significancia: p < {args.p_threshold}")
    print("==========================================================================")
    
    for b_name in target_bands:
        if b_name not in band_names:
            print(f"[AVISO] Banda desconocida: {b_name}, omitiendo.")
            continue
            
        b_idx = band_names.index(b_name)
        p_band = p_values[:, :, b_idx, :]  # (dest, src, epoch)
        
        # Cargar t-values correspondientes si existen
        t_candidates = [
            args.p_file.replace("_naive.npy", "_t_values.npy"),
            args.p_file.replace(".npy", "_t_values.npy"),
            f"plots/p_values_rois_{args.roi_method}_t_values.npy",
            "plots/p_values_rois_t_values.npy"
        ]
        t_band = None
        for cand in t_candidates:
            if os.path.exists(cand):
                try:
                    t_all = np.load(cand)
                    if t_all.shape == p_values.shape:
                        t_band = t_all[:, :, b_idx, :]
                        break
                except Exception:
                    pass
        
        out_html = os.path.join(args.output_dir, f"red_temporal_{b_name}_{args.roi_method}.html")
        try:
            export_interactive_temporal_3d_network(
                coords_3d=coords_3d,
                p_values_band=p_band,
                node_names=node_names,
                band_name=b_name,
                filename=out_html,
                p_threshold=args.p_threshold,
                epoch_duration=parameters.EPOCH_DURATION_S,
                t_values_band=t_band
            )
            if args.roi_method == "mean":
                # Guardar copia base sin sufijo
                base_html = os.path.join(args.output_dir, f"red_temporal_{b_name}.html")
                if base_html != out_html:
                    import shutil
                    shutil.copyfile(out_html, base_html)
            print(f"  -> Archivo HTML interactivo ({args.roi_method.upper()}): {out_html}")
        except ImportError:
            print("\n[ERROR DE DEPENDENCIA]")
            print("Plotly no está instalado en este entorno de Python.")
            print("Para solucionarlo, ejecuta en tu terminal:")
            print("    pip install plotly")
            print("o si estás en un entorno virtual:")
            print("    source .venv/bin/activate")
            sys.exit(1)
            
    print("\n¡Visualizaciones temporales generadas exitosamente!")
    print("Puedes abrirlas en tu navegador (doble clic en el archivo .html) para rotar en 3D")
    print("y mover la barra de tiempo para ver la habituación época a época.")

if __name__ == "__main__":
    main()

