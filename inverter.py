import numpy as np

def D_hat(K, J, h):
    mesh = np.meshgrid(range(J), range(K))
    return np.array([1j/h * np.sin(2 * np.pi * mesh[0].T / J),
                     1j/h * np.sin(2 * np.pi * mesh[1].T / K)])

def L_hat(K, J, h):
    mesh = np.meshgrid(range(J), range(K))
    return -4/(h*h) * (np.square(np.sin(np.pi * mesh[0].T / J)) + np.square(np.sin(np.pi * mesh[1].T / K)))

def solver(w, dt, rho, mu, K, J, h):
    w_hat = np.array([np.fft.fft2(w[0, ...]), np.fft.fft2(w[0, ...])])
    D = D_hat(K, J, h)
    dotp = D[0, ...] * w_hat[0, ...] + D[1, ...] * w_hat[1, ...]
    d_square = np.linalg.norm(D, axis = 0) * np.linalg.norm(D, axis = 0)
        
    # risky ell:
    ell_risk = [0]
    if J % 2 == 0:
        ell_risk.append(J//2)
        
    em_risk = [0]
    if K % 2 == 0:
        em_risk.append(K//2)
    
    for l in ell_risk:
        for m in em_risk:
            d_square[l, m] = 1
        
    q_hat = dotp / (d_square * dt / rho)
    denom_2 = (1 + dt * mu / (2*rho) * L_hat(K, J, h))
    u0_hat = (w_hat[0, ...] - D[0, ...] * dotp / d_square) / denom_2
    u1_hat = (w_hat[1, ...] - D[1, ...] * dotp / d_square) / denom_2

    for l in ell_risk:
        for m in em_risk:
            u0_hat[l, m] = w_hat[0, l, m] / denom_2[l, m]
            u1_hat[l, m] = w_hat[1, l, m] / denom_2[l, m]
            
    return np.real(np.array([np.fft.ifft2(u0_hat), np.fft.ifft2(u1_hat)]))

        
    
        
        
    # deal with div by zero:
    
    
    