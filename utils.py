import numpy as np
from statsmodels.tsa.api import VAR


# Metodo AIC para encontrar p
def AIC(data, n_epochs, m, n):
    lags = range(1, 21)
    AIC = np.zeros(20)


    for epoch in range(n_epochs):
        data_epoch = data[epoch].T
        model = VAR(data_epoch)
        
        print(f"Epoca: {epoch}")
        for lag in lags:
            print(f"Lag: {lag}")

            _, log_det_v = np.linalg.slogdet(model.fit(maxlags=lag).sigma_u) # sign, log_det_v

            # AIC[p] = 2k - 2 ln(L_hat)
            # k = p * m**2 es el n. de parametros
            # L_hat ≈ det(V) donde V es la matriz de covarianza del error (gaussiano)
            aic = log_det_v + (2 * lag * (m**2)) / n
            AIC[lag - 1] += aic / n_epochs
    
    return lags[np.argmin(AIC)]


# Calculo manual, para cada epoca, de 
def V_epoch(data, A, p, epoch, n_nodes, n_samples):
    data_epoch = data[epoch]

    X_hats = np.zeros((n_nodes, n_samples - p))
    
    for k in range(p):
        X_prevs = data_epoch[:, p - (k + 1) : n_samples - (k + 1)]
        X_hats += A[epoch][k] @ X_prevs

    Xs = data_epoch[:, p:]

    errors = Xs - X_hats
    return np.cov(errors)
