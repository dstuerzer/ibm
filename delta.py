import numpy as np

def delta_x(rr, h):
    r = rr / h
    q = (3 - 2 * r + np.sqrt(1 + 4 * r - 4 * r * r)) / 8
    z = np.zeros((4,4))
    z[3, :] = 0.5 - q
    z[2, :] = (4 * r - 2) / 8 + q
    z[1, :] = q
    z[0, :] = (6 - 4 * r) / 8 - q
    return z / h

def delta_y(r, h):
    return delta_x(r, h).T