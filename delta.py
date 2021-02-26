import numpy as np

def fun01(r):
    return (3 - 2 * r + np.sqrt(1 + 4 * r - 4 * r * r)) / 8.0

def delta(r):
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
    
def delta_h(r,h):
    return 1/h * delta(r / h)