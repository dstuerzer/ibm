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
    bending_K = 0.1
    elong = 0.125
    freq = 1
    
    Z_0 = np.array([0.5, 0.75 + elong * np.sin(freq * t * 2 * np.pi)])
    
    N = X.shape[0]
    
    norms = [np.linalg.norm(X[s+1, :] - X[s, :]) for s in range(N-1)]
    T = [tension_K* (norms[s] / dq - 1) for s in range(N-1)]
    tau = [(X[s+1, :] - X[s, :]) / norms[s] for s in range(N-1)]
    F_tension = [(T[0] * tau[0])/ dq]
    F_tension += [(T[s] * tau[s] - T[s-1] * tau[s-1]) / dq for s in range(1, N-1)]
    F_tension += [(- T[N-2] * tau[N-2]) / dq]
    
    F_tether = [-tether_K * (X[0, :] - Z_0)]
    F_tether += [np.array([0,0]) for _ in range(N-1)]
    
    Ck = (X[2:, :] + X[:-2, :] - 2 * X[1:-1, :]) / (dq * dq)
    C_ex = np.array([np.array([0, 0]), np.array([0, 0])] + list(Ck) + [np.array([0, 0]), np.array([0, 0])])
    F_bending =  -bending_K / (dq * dq) * (C_ex[2:, :] + C_ex[:-2, :] - 2 * C_ex[1:-1, :])
    
    return np.array(F_tension) + np.array(F_tether) + np.array(F_bending), Z_0 - X[0, :]
 
   