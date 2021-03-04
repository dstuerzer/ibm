import numpy as np
import operators as op

def D_hat(J, K, h):
    mesh = np.meshgrid(range(J), range(K))
    return np.array([1j/h * np.sin(2 * np.pi * mesh[0].T / J),
                     1j/h * np.sin(2 * np.pi * mesh[1].T / K)])

def L_hat(J, K, h):
    mesh = np.meshgrid(range(J), range(K))
    return -4/(h*h) * (np.square(np.sin(np.pi * mesh[0].T / J)) + np.square(np.sin(np.pi * mesh[1].T / K)))

def solver(w, dt, rho, mu, J, K, h):
    w_hat = np.array([np.fft.fft2(w[0, ...]), np.fft.fft2(w[1, ...])])
    D = D_hat(J, K, h)
    d_dot_w = op.vdot(D, w_hat)
    d_dot_d = op.vdot(D, D)
        
    # risky indices:
    ell_risk = [0]
    if J % 2 == 0:
        ell_risk.append(J//2)
        
    em_risk = [0]
    if K % 2 == 0:
        em_risk.append(K//2)
    
    for l in ell_risk:
        for m in em_risk:
            d_dot_d[l, m] = 1
        
    # continue here
    denom_2 = (1 - dt * mu / (2*rho) * L_hat(J, K, h))
    u0_hat = (w_hat[0, ...] - D[0, ...] * d_dot_w / d_dot_d) / denom_2
    u1_hat = (w_hat[1, ...] - D[1, ...] * d_dot_w / d_dot_d) / denom_2

    for l in ell_risk:
        for m in em_risk:
            u0_hat[l, m] = w_hat[0, l, m] / denom_2[l, m]
            u1_hat[l, m] = w_hat[1, l, m] / denom_2[l, m]
            
    return np.real(np.array([np.fft.ifft2(u0_hat), np.fft.ifft2(u1_hat)]))
