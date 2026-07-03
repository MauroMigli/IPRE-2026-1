import numpy as np
import os
import parameters

def main():
    file_path = "plots/p_values_welch_naive.npy"
    if not os.path.exists(file_path):
        print(f"Error: No se encontró el archivo '{file_path}'.")
        return

    # Cargar matriz (dest, src, band, epoch)
    p_values_4d = np.load(file_path)
    n_dest, n_src, n_band, n_epoch = p_values_4d.shape
    
    bands = list(parameters.F_BANDS.keys())
    
    # Calcular el número de conexiones posibles (excluyendo diagonal y NaNs)
    # Tomamos la primera época/banda como referencia para contar las conexiones válidas
    ref_matrix = p_values_4d[:, :, 0, 0]
    total_possible = np.sum(~np.isnan(ref_matrix)) - np.sum(~np.isnan(np.diag(ref_matrix)))
    
    print("======================================================================")
    print(" IPRE-2026: Conteo de Conexiones Significativas (Welch Naive)")
    print("======================================================================")
    print(f"Número de canales activos: {n_dest}")
    print(f"Conexiones evaluadas por época (sin diagonal): {total_possible}")
    print(f"Número de épocas: {n_epoch}")
    print("----------------------------------------------------------------------")
    print(f"{'Banda':<10} | {'p < 0.05 (Esperado: ' + f'{total_possible*0.05:.1f}':<25} | {'p < 0.01 (Esperado: ' + f'{total_possible*0.01:.1f}':<25}")
    print("----------------------------------------------------------------------")

    for b_idx, band_name in enumerate(bands):
        counts_05 = []
        counts_01 = []
        
        for ep in range(n_epoch):
            p_matrix = p_values_4d[:, :, b_idx, ep].copy()
            # Llenar diagonal con NaN para no contarla
            np.fill_diagonal(p_matrix, np.nan)
            
            # Contar conexiones significativas
            n_sig_05 = np.sum(p_matrix < 0.05)
            n_sig_01 = np.sum(p_matrix < 0.01)
            
            counts_05.append(n_sig_05)
            counts_01.append(n_sig_01)
            
        mean_05 = np.mean(counts_05)
        std_05 = np.std(counts_05)
        
        mean_01 = np.mean(counts_01)
        std_01 = np.std(counts_01)
        
        print(f"{band_name:<10} | Promedio: {mean_05:>5.1f} ± {std_05:<4.1f} ({mean_05/total_possible*100:>5.1f}%) | Promedio: {mean_01:>5.1f} ± {std_01:<4.1f} ({mean_01/total_possible*100:>5.1f}%)")
    print("======================================================================")

if __name__ == "__main__":
    main()
