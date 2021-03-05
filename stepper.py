import numpy as np
import operators as op
from delta import delta_x, delta_y, delta_h
import inverter as inv
from matplotlib import pyplot as plt
from time import time

def first_integration(UV, XY, h, K, J):
    # UV is full 2 x J x K velocity
    # XY is X(s)^n, a 2-vector
    j0, k0 = int(XY[0]/h), int(XY[1]/h)
    rx, ry = XY[0] - h * j0, XY[1] - h * k0
    phi = delta_x(rx, h) * delta_y(ry, h)

    jts = [j % J for j in range(j0-1, j0+3)]
    ks  = [k % K for k in range(k0-1, k0+3)]
    ux_local = np.take(np.take(UV[0, ...], jts, axis=0), ks, axis=1)
    uy_local = np.take(np.take(UV[1, ...], jts, axis=0), ks, axis=1)
    return np.array([np.sum(ux_local * phi), np.sum(uy_local * phi)])


def first_integration_old(UV, XY, h, K, J):
    # UV is full 2 x J x K velocity
    # XY is X(s)^n, a 2-vector
    j0, k0 = int(XY[0]/h), int(XY[1]/h)
    integral = np.zeros((2,))
    for j in range(j0-3, j0+4):
        for k in range(k0 - 3, k0+4):
            _j, _k = j % J, k % K  # <- for u
            xj, yk = j * h, k * h  # might be "out of bounds"
            integral += UV[:, _j, _k] * delta_h(xj - XY[0], h) * delta_h(yk - XY[1], h)

    return integral

def second_integration(FF, XY, h, J, K, d_theta):
    # FF == F1
    # XY == X1, our curve
    f1 = np.zeros((2, J, K))


    for s in range(FF.shape[0]):
        j0, k0 = int(XY[s, 0]/h), int(XY[s, 1]/h)
        rx, ry = XY[s, 0] - h * j0, XY[s, 1] - h * k0
        phi_x = FF[s, 0] * delta_x(rx, h) * delta_y(ry, h) * d_theta
        phi_y = FF[s, 1] * delta_x(rx, h) * delta_y(ry, h) * d_theta

        jts = [j % J for j in range(j0-1, j0+3)]
        ks = [k % K for k in range(k0-1, k0+3)]
        f1[0][np.ix_(jts, ks)] += phi_x
        f1[1][np.ix_(jts, ks)] += phi_y

    return f1


def second_integration_old(FF, XY, h, J, K, d_theta):
    # FF == F1
    # XY == X1, our curve
    f1 = np.zeros((2, J, K))
    for s in range(FF.shape[0]):
        j0, k0 = int(XY[s, 0]/h), int(XY[s, 1]/h)
        for j in range(j0-3, j0+4):
            for k in range(k0 - 3, k0+4):
                _j, _k = j % J, k % K  # <- for u
                xj, yk = j * h, k * h
                f1[:, _j, _k] += FF[s, :] * delta_h(xj - XY[s, 0], h) * delta_h(yk - XY[s, 1], h) * d_theta
    return f1


def RK(X, u0, dt, h, K, J, _tension_K, d_theta, N_theta, _rho, _mu):

    X1 = X.copy()
    for s in range(N_theta):
        X1[s, :] += dt * h * h * first_integration(u0, X[s, :], h, K, J) * 0.5

    # >>>>>>>>>>>
 #   X1_old = X.copy()
  #  for s in range(N_theta):
   #     X1_old[s, :] += dt * h * h * first_integration_old(u0, X[s, :], h, K, J) * 0.5

   # print(np.linalg.norm(X1 - X1_old))
    # <<<<<<<<<<<<<

    F1 = _tension_K/(d_theta * d_theta) * (np.roll(X1, 1, axis=0) + np.roll(X1, -1, axis=0) - 2 * X1)

    f1 = second_integration(F1, X1, h, J, K, d_theta)
    
    # >>>>>>>>>
 #   f1_old = second_integration_old(F1, X1, h, J, K, d_theta)
  #  
   # print(np.linalg.norm(f1 - f1_old))
    # <<<<<<<<<<
    
    w1 = u0 - dt / 2 * op.Suu(u0, h) + dt / (2 * _rho) * f1

    u1 = inv.solver(w1, dt, _rho, _mu, J, K, h)

    X2 = X.copy()
    for s in range(N_theta):
        X2[s, :] += dt * h * h * first_integration(u1, X1[s, :], h, K, J)

    w2 = u0 - dt * op.Suu(u1, h) + dt / _rho * f1 + dt * _mu / (2 * _rho) * op.L2(u0, h)

    u2 = inv.solver(w2, dt, _rho, _mu, J, K, h)

    return X2, u2






