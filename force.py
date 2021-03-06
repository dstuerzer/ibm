import numpy as np

# =============================================================================
# def Force(X, dq):
#     _tension_K = 1
#     return _tension_K/(dq * dq) * (np.roll(X, 1, axis=0) + np.roll(X, -1, axis=0) - 2 * X)
#  
# =============================================================================


def Force(X, dq, t=0):
    tension_K = 1
    tether_K = 50
    elong = dq * 5.0
    freq = 2
    
    Z_0 = np.array([0.5, 2.5])
    Z_1 = np.array([0.5 + elong - dq, 2.5 + elong * np.sin(freq * t)])
    
    N = X.shape[0]
    
    norms = [np.linalg.norm(X[s+1, :] - X[s, :]) for s in range(N-1)]
    T = [tension_K* (norms[s] / dq - 1) for s in range(N-1)]
    tau = [(X[s+1, :] - X[s, :]) / norms[s] for s in range(N-1)]
    F_tension = [(T[0] * tau[0])/ dq]
    F_tension += [(T[s] * tau[s] - T[s-1] * tau[s-1]) / dq for s in range(1, N-1)]
    F_tension += [(- T[N-2] * tau[N-2]) / dq]
    
    F_tether = [-tether_K * (X[0, :] - Z_0),
                np.array([0,0]), np.array([0,0]),
                np.array([0,0]), np.array([0,0]),
                -tether_K * (X[5, :] - Z_1)]
    F_tether += [np.array([0,0]) for _ in range(N-6)]
    
    return np.array(F_tension) + np.array(F_tether)
 
   