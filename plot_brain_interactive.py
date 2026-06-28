import numpy as np
import os
import argparse
import parameters
import html_plotter

def main():
    parser = argparse.ArgumentParser(description="Generar visualización 3D interactiva con slider temporal para una banda específica.")
    parser.add_argument("--band", type=str, required=True, choices=list(parameters.F_BANDS.keys()), 
                        help="Nombre de la banda de frecuencia (Delta, Theta, Alpha, Beta, Gamma)")
    parser.add_argument("--adjacency", type=str, choices=['total', 'null'], default='total',
                        help="Tipo de matriz de adyacencia espacial a utilizar (total o null).")
    parser.add_argument("--output", type=str, default=None, 
                        help="Ruta del archivo HTML de salida (por defecto auto-generado en la carpeta plots)")
    
    args = parser.parse_args()
    
    # 1. Definir input en base a la adyacencia
    input_file = f"plots/p_values_empiricos_{args.adjacency}.npy"
    
    if not os.path.exists(input_file):
        fallback_file = "plots/p_values_empiricos.npy"
        if os.path.exists(fallback_file):
            print(f"Advertencia: No se encontró '{input_file}'. Usando archivo de respaldo '{fallback_file}'.")
            input_file = fallback_file
        else:
            print(f"Error: No se encontró '{input_file}' ni '{fallback_file}'.")
            print("Asegúrate de que el pipeline principal haya terminado y guardado la matriz 4D.")
            return
        
    print(f"Cargando matriz de p-valores ({args.adjacency}) desde '{input_file}'...")
    p_values_4d = np.load(input_file) # shape: (dest, src, band, epoch)
    
    n_dest, n_src, n_band, n_epoch = p_values_4d.shape
    bands_list = list(parameters.F_BANDS.keys())
    band_idx = bands_list.index(args.band)
    
    # 2. Cargar nombres de canales activos
    ch_names_path = "plots/channel_names.npy"
    if os.path.exists(ch_names_path):
        channel_names = np.load(ch_names_path, allow_pickle=True)
    else:
        print(f"Error: No se encontró el archivo '{ch_names_path}' con los nombres de los canales.")
        return
    
    # 3. Extraer el tensor 3D para la banda seleccionada (dest, src, epoch)
    p_values_3d = p_values_4d[:, :, band_idx, :]
    
    # 4. Definir nombre de salida
    if args.output is None:
        filename = f"plots/cerebro_evolucion_{args.band}_{args.adjacency}.html"
    else:
        filename = args.output
        
    # 5. Obtener posiciones 3D de los electrodos
    coords_3d = html_plotter.get_3d_positions(parameters.ELP_FILE, channel_names)
    
    # 6. Graficar con slider
    html_plotter.export_slider_3d_network(
        coords_3d,
        p_values_3d,
        channel_names,
        filename=filename,
        dropped_channels=parameters.DROPPED_CHANNELS,
        hide_isolated_nodes=False
    )

if __name__ == "__main__":
    main()
