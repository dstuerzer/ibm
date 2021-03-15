import numpy as np
import operators as op
from delta import delta_x, delta_y, delta_z
import inverter as inv
from force import Force
from time import time
from integrations import first_integration, second_integration



def RK(X, u0, dt, h, K, J, d_theta, N_theta, _rho, _mu, dict_of_targets):

    tt = time()
    X1 = X.copy()
    for s in range(N_theta):
        X1[s, :] += dt * h * h * first_integration(u0, X[s, :], h, K, J) * 0.5

    print("first    ", "{:.1f} ms".format(1000 * (time() - tt)))
    tt = time()
    
    F1 = Force(X1, d_theta, dict_of_targets)

    print("force    ", "{:.1f} ms".format(1000 * (time() - tt)))
    tt = time()
    
    f1 = second_integration(F1, X1, h, J, K, d_theta)
        
    print("second   ", "{:.1f} ms".format(1000 * (time() - tt)))
    tt = time()
    
    # gravity 
    f1[0, ...] = f1[0, ...] + 0.1

    w1 = u0 - dt / 2 * op.Suu(u0, h) + dt / (2 * _rho) * f1

    print("w        ", "{:.1f} ms".format(1000 * (time() - tt)))
    tt = time()
    
    u1 = inv.solver(w1, dt, _rho, _mu, J, K, h)

    print("solver   ", "{:.1f} ms".format(1000 * (time() - tt)))
    tt = time()
    
    X2 = X.copy()
    for s in range(N_theta):
        X2[s, :] += dt * h * h * first_integration(u1, X1[s, :], h, K, J)

    print("first    ", "{:.1f} ms".format(1000 * (time() - tt)))
    tt = time()
    
    w2 = u0 - dt * op.Suu(u1, h) + dt / _rho * f1 + dt * _mu / (2 * _rho) * op.L2(u0, h)

    print("w2       ", "{:.1f} ms".format(1000 * (time() - tt)))
    tt = time()
    
    u2 = inv.solver(w2, dt, _rho, _mu, J, K, h)

    print("sovler   ", "{:.1f} ms".format(1000 * (time() - tt)))
    tt = time()
    
    return X2, u2






