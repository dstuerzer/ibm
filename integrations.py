import numpy as np
from delta import delta_x, delta_y, delta_z
from time import time

# from delta import delta_x, delta_y, delta_z

def first_integration(UV, XY, h, K, J):
    # UV is full 2 x J x K velocity
    # XY is X(s)^n, a 2-vector
    j0, k0 = int(XY[0]/h), int(XY[1]/h)
    rx, ry = XY[0] - h * j0, XY[1] - h * k0
  
    zx = delta_z(rx, h)
    zy = delta_z(ry, h)

    jts = [j % J for j in range(j0-1, j0+3)]
    ks  = [k % K for k in range(k0-1, k0+3)]
    sumx, sumy = 0, 0
    _j = 0
    for j in jts:
        _k = 0
        for k in ks:
            phi_j_k = zx[_j] * zy[_k] / (h * h)
            sumx += UV[0, j, k] * phi_j_k
            sumy += UV[1, j, k] * phi_j_k
            _k += 1
        _j += 1
     
    return np.array([sumx, sumy])



def second_integration(FF, XY, h, J, K, d_theta):
    # FF == F1
    # XY == X1, our curve
    f1 = np.zeros((2, J, K))
    
    for s in range(FF.shape[0]):
        j0, k0 = int(XY[s, 0]/h), int(XY[s, 1]/h)
        rx, ry = XY[s, 0] - h * j0, XY[s, 1] - h * k0
        zx = delta_z(rx, h)
        zy = delta_z(ry, h)

        _j = 0
        for jj in [j % J for j in range(j0-1, j0+3)]:
            _k = 0
            for kk in [k % K for k in range(k0-1, k0+3)]:
                phi_j_k = zx[_j] * zy[_k] * d_theta / (h * h)
                f1[0][jj, kk] += FF[s, 0] *  phi_j_k
                f1[1][jj, kk] += FF[s, 1] *  phi_j_k
                _k += 1
            _j += 1
    return f1

