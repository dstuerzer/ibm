import numpy as np

# =============================================================================
# def Force(X, dq):
#     _tension_K = 1
#     return _tension_K/(dq * dq) * (np.roll(X, 1, axis=0) + np.roll(X, -1, axis=0) - 2 * X)
#  
# =============================================================================


def F_tension(X, dq):
    tension_K = 1
    N = X.shape[0]
    
    norms = [np.linalg.norm(X[s+1, :] - X[s, :]) for s in range(N-1)]
    T = [tension_K* (norms[s] / dq - 1) for s in range(N-1)]
    tau = [(X[s+1, :] - X[s, :]) / norms[s] for s in range(N-1)]
    f_tension = [(T[0] * tau[0])/ dq]
    f_tension += [(T[s] * tau[s] - T[s-1] * tau[s-1]) / dq for s in range(1, N-1)]
    f_tension += [(- T[N-2] * tau[N-2]) / dq]
    return np.array(f_tension)

def F_bending(X, dq):
    bending_K = 0.1
    Ck = (X[2:, :] + X[:-2, :] - 2 * X[1:-1, :]) / (dq * dq)
    C_ex = np.array([np.array([0, 0]), np.array([0, 0])] + list(Ck) + [np.array([0, 0]), np.array([0, 0])])
    return np.array(-bending_K / (dq * dq) * (C_ex[2:, :] + C_ex[:-2, :] - 2 * C_ex[1:-1, :]))
    
def F_tether(X, dq, dict_of_targets):
    tether_K = 50
    f_tether = [np.array([0, 0]) for _ in range(X.shape[0])]
    for i, Z in dict_of_targets.items():
        f_tether[i] = -tether_K * (X[i, :] - Z)
        
    return np.array(f_tether)
    
def Force(X, dq, dict_of_targets=None):
    total_force = np.zeros(X.shape)
    total_force += F_tension(X, dq)
    total_force += F_bending(X, dq)
    if dict_of_targets is not None:
        total_force += F_tether(X, dq, dict_of_targets)
        
    return total_force
 
   