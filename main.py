import numpy as np
import operators as op
from delta import delta_h

def printu(u):
    print(u.T[::-1,:])
    
# def xy(i, j):
 #   return (i % N_x) * h, (j % N_y) * h
    
# set up grid
h = 1
x_max = 4
y_max = 5

J = int(x_max / h)
h = x_max / J
K = int(y_max / h)

# set up boundary
N_theta = 3
d_theta = 2 * np.pi / N_theta
X = np.array([np.array([np.cos(s * d_theta) + J * h/2, np.sin(s * d_theta) + K /2]) for s in range(N_theta)])


# setup initial conditions
u = np.zeros((2, J, K))  # must be divergence-free
u = np.random.normal(0, 1, size=(2, J, K))

# parameters

_tension_K = 1
_mu = 1
_rho = 1
dt = 1


# prelim step


def first_integration(UV, XY, h):
    # UV is full 2 x J x K velocity
    # XY is X(s)^n, a 2-vector
    j0, k0 = int(XY[0]/h), int(XY[1]/h)
    integral = np.zeros((2,))
    for j in range(j0-3, j0+4):
        for k in range(k0 - 3, k0+4):
            _j, _k = j % J, k % K  # <- for u
            xj, yk = j * h, k * h
            integral += UV[:, _j, _k] * delta_h(xj - XY[0], h) * delta_h(yk - XY[1], h)
            
    return integral

X1 = X.copy()
for s in range(N_theta):
    X1[s, :] += dt * h * h * first_integration(u, X[s, :], h)

F1 = _tension_K/(d_theta * d_theta) * (np.roll(X1, 1, axis=0) + np.roll(X1, -1, axis=0) - 2 * X1)


def second_integration(FF, XY, h, J, K):
    f1 = np.zeros((2, J, K))
    for s in range(FF.shape[0]):
        j0, k0 = int(XY[0]/h), int(XY[1]/h)
        inner_sum = np.zeros((2,))
        for j in range(j0-3, j0+4):
            for k in range(k0 - 3, k0+4):
                _j, _k = j % J, k % K  # <- for u
                xj, yk = j * h, k * h
                f1[:, _j, _k] += FF[s, :]
                
                
                
                
                
                
                
                
                
                
                
                
                
                