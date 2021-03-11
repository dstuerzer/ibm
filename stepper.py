import numpy as np
import operators as op
from delta import delta_x, delta_y, delta_z
import inverter as inv
from force import Force


def first_integration(UV, XY, h, K, J):
    # UV is full 2 x J x K velocity
    # XY is X(s)^n, a 2-vector
    j0, k0 = int(XY[0]/h), int(XY[1]/h)
    rx, ry = XY[0] - h * j0, XY[1] - h * k0

    zx = delta_z(rx, h)
    zy = delta_z(ry, h)
    phi = np.zeros((4,4))
    for j in range(4):
        for k in range(4):
            phi[j,k] = zx[j] * zy[k] / (h * h)

    jts = [j % J for j in range(j0-1, j0+3)]
    ks  = [k % K for k in range(k0-1, k0+3)]

    ux_local = np.zeros((4,4))
    uy_local = np.zeros((4,4))
    _j = 0
    for j in jts:
        _k = 0
        for k in ks:
            ux_local[_j, _k] = UV[0, j, k]
            uy_local[_j, _k] = UV[1, j, k]
            _k += 1
        _j += 1
    return np.array([np.sum(ux_local * phi), np.sum(uy_local * phi)])


def second_integration(FF, XY, h, J, K, d_theta):
    # FF == F1
    # XY == X1, our curve
    f1 = np.zeros((2, J, K))
    for s in range(FF.shape[0]):
        j0, k0 = int(XY[s, 0]/h), int(XY[s, 1]/h)
        rx, ry = XY[s, 0] - h * j0, XY[s, 1] - h * k0
        phi_x = FF[s, 0] * delta_x(rx, h) * delta_y(ry, h) * d_theta
        phi_y = FF[s, 1] * delta_x(rx, h) * delta_y(ry, h) * d_theta
        _j = 0
        for jj in [j % J for j in range(j0-1, j0+3)]:
            _k = 0
            for kk in [k % K for k in range(k0-1, k0+3)]:
                f1[0][jj, kk] += phi_x[_j, _k]
                f1[1][jj, kk] += phi_y[_j, _k]
                _k += 1
            _j += 1

    return f1


def RK(X, u0, dt, h, K, J, d_theta, N_theta, _rho, _mu, dict_of_targets):

    X1 = X.copy()
    for s in range(N_theta):
        X1[s, :] += dt * h * h * first_integration(u0, X[s, :], h, K, J) * 0.5

    F1 = Force(X1, d_theta, dict_of_targets)

    f1 = second_integration(F1, X1, h, J, K, d_theta)
    
    # gravity 
    f1[0, ...] = f1[0, ...] + 0.1

    w1 = u0 - dt / 2 * op.Suu(u0, h) + dt / (2 * _rho) * f1

    u1 = inv.solver(w1, dt, _rho, _mu, J, K, h)

    X2 = X.copy()
    for s in range(N_theta):
        X2[s, :] += dt * h * h * first_integration(u1, X1[s, :], h, K, J)

    w2 = u0 - dt * op.Suu(u1, h) + dt / _rho * f1 + dt * _mu / (2 * _rho) * op.L2(u0, h)

    u2 = inv.solver(w2, dt, _rho, _mu, J, K, h)

    return X2, u2






