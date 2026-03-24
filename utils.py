import numpy as np
from statsmodels.tsa.api import VAR

#################################
###### REQUIERE ACTUALIZAR ######
#################################


def AIC_BIC(data, n_epochs, m, T) -> tuple:
    lags = range(1, 21)
    AIC = np.zeros(20)
    BIC = np.zeros(20)
    

    for epoch in range(n_epochs):
        data_epoch = data[epoch].T
        model = VAR(data_epoch)
        
        for lag in lags:

            _, log_det_v = np.linalg.slogdet(model.fit(maxlags=lag).sigma_u) # sign, log_det_v

            # AIC[p] = 2k + 2 ln(L_hat)
            # k = p * m**2 es el n. de parametros
            # L_hat ≈ 1/det(V) donde V es la matriz de covarianza del error (gaussiano)
            aic = log_det_v * T + (2 * lag * (m**2))
            bic = log_det_v * T + np.log(T) * lag * (m**2)
            AIC[lag - 1] += aic / n_epochs
            BIC[lag - 1] += bic / n_epochs
    
    return lags[np.argmin(AIC)], lags[np.argmin(BIC)]


# Calculo manual, para cada epoca, de 
def V_epoch(data_epoch, A_epoch, p, n_nodes, n_samples):
    

    X_hats = np.zeros((n_nodes, n_samples - p))
    
    for k in range(p):
        X_prevs = data_epoch[:, p - (k + 1) : n_samples - (k + 1)]
        X_hats += A_epoch[k] @ X_prevs

    Xs = data_epoch[:, p:]

    errors = Xs - X_hats
    return np.cov(errors)
