import numpy as np

# u: RxR -> RxR
# phi: RxR -> R

# roll(1, axis=0): (j, k) -> (j+1, k)
# roll(1, axis=1): (j, k) -> (j, k+1)

def Dx(phi, h):
    return (np.roll(phi, -1, axis = 0) - np.roll(phi, 1, axis = 0)) / (2 * h)

def Dy(phi, h):
    return (np.roll(phi, -1, axis = 1) - np.roll(phi, 1, axis = 1)) / (2 * h)

def grad(phi, h):
    return np.array([Dx(phi, h), Dy(phi, h)])

def Div(u, h):
    return Dx(u[0,...], h) + Dy(u[1, ...], h)

def L(phi, h):
    ell = np.roll(phi, -1, axis = 0) + np.roll(phi, 1, axis = 0) - 2 * phi
    ell += np.roll(phi, -1, axis = 1) + np.roll(phi, 1, axis = 1) - 2 * phi
    return ell / (h*h)

def L2(u, h):
    return np.array([L(u[0, ...], h), L(u[1, ...], h)])

def S(u, phi, h):
    return 0.5 * vdot(u, grad(phi, h)) + 0.5 * Div(u * phi, h)
    
def Suu(u, h):
    return np.array([S(u, u[0, ...], h), S(u, u[1, ...], h)])
    
def vdot(u, v):
    return np.sum(u * v, axis = 0)