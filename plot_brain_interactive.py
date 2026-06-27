import numpy as np
import os
import argparse
import parameters
import html_plotter

def main():
    parser = argparse.ArgumentParser(description="Generar visualización 3D interactiva para una época y banda específicas.")
    parser.add_argument("--epoch", type=int, required=True, help="Índice de la época a graficar (ej. 5)")
    parser.add_argument("--band", type=str, required=True, choices=list(parameters.F_BANDS.keys()), 
                        help="Nombre de la banda de frecuencia (Delta, Theta, Alpha, Beta, Gamma)")
    parser.add_argument("--input", type=str, default="plots/p_values_empiricos.npy", 
                        help="Ruta al archivo .npy con la matriz 4D de p-valores")
    parser.add_argument("--output", type=str, default=None, 
                        help="Ruta del archivo HTML de salida (por defecto auto-generado en la carpeta plots)")
    
    args = parser.parse_args()
    
    # 1. Cargar la matriz 4D
    if not os.path.exists(args.input):
        print(f"Error: No se encontró el archivo de entrada '{args.input}'.")
        print("Asegúrate de que el pipeline principal haya terminado y guardado la matriz 4D.")
        return
        
    print(f"Cargando matriz de p-valores desde '{args.input}'...")
    p_values_4d = np.load(args.input) # shape: (dest, src, band, epoch)
    
    n_dest, n_src, n_band, n_epoch = p_values_4d.shape
    bands_list = list(parameters.F_BANDS.keys())
    
    # Validar argumentos
    if args.epoch < 0 or args.epoch >= n_epoch:
        print(f"Error: La época debe estar entre 0 y {n_epoch - 1}. Recibido: {args.epoch}")
        return
        
    band_idx = bands_list.index(args.band)
    
    # 2. Cargar nombres de canales activos (descartando los eliminados en parameters)
    # Reconstruimos la lista de canales que usó test_tfce.py
    # Para esto, leemos un archivo .set de referencia o usamos los parámetros.
    # Dado que test_tfce.py usa el orden de canales del primer archivo que procesa, 
    # podemos extraer los nombres de canales directamente si guardamos un archivo de mapeo o nombres.
    # Por simplicidad, intentaremos leer el orden de canales guardado en el pipeline.
    ch_names_path = "plots/channel_names.npy"
    if os.path.exists(ch_names_path):
        channel_names = np.load(ch_names_path, allow_pickle=True)
    else:
        print(f"Error: No se encontró el archivo '{ch_names_path}' con los nombres de los canales.")
        return
    
    # 3. Extraer la rebanada 2D para la época y banda seleccionadas
    # shape de la rebanada: (dest, src)
    p_values_2d = p_values_4d[:, :, band_idx, args.epoch]
    
    # 4. Definir nombre de salida
    if args.output is None:
        filename = f"plots/cerebro_3d_epoch{args.epoch}_{args.band}.html"
    else:
        filename = args.output
        
    # 5. Obtener posiciones 3D de los electrodos
    coords_3d = html_plotter.get_3d_positions(parameters.ELP_FILE, channel_names)
    
    # 6. Graficar
    html_plotter.export_interactive_3d_network(
        coords_3d,
        p_values_2d,
        channel_names,
        filename=filename,
        dropped_channels=parameters.DROPPED_CHANNELS,
        hide_isolated_nodes=False
    )
    
    print(f"¡Listo! Visualización 3D guardada en: {filename}")

if __name__ == "__main__":
    main()
