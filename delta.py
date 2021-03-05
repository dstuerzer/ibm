import numpy as np

def fun01(r):
    return (3 - 2 * r + np.sqrt(1 + 4 * r - 4 * r * r)) / 8.0

def dlta(r):
    if r < -2.0:
        return 0
    elif r < -1:
        return 0.5 - fun01(r + 2)
    elif r < 0:
        return (4 * r + 2) / 8 + fun01(r + 1)
    elif r < 1:
        return fun01(r)
    elif r < 2:
        return (10 - 4 * r) / 8 - fun01(r - 1)
    else:
        return 0
    
def delta_h(x,h):
    return dlta(x / h) / h

def delta_x(r, h):
    q = (3 - 2 * r + np.sqrt(1 + 4 * r - 4 * r * r)) / 8.0
    z = np.zeros((4,4))
    z[3, :] = 0.5 - q
    z[2, :] = (4 * r + 2) / 8 + q
    z[1, :] = q
    z[0, :] = (10 - 4 * r) / 8 - q
    return z

def delta_y(r, h):
    return delta_x(r, h)