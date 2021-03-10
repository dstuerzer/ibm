import numpy as np
import operators as op
import stepper as st
from matplotlib import pyplot as plt
import delta
from time import time

 # %matplotlib qt
       
def printu(u):
    return u.T[::-1, :]

# def xy(i, j):
 #   return (i % N_x) * h, (j % N_y) * h
    
# set up grid
plotting = True

# parameters

_mu = 0.1
_rho = 0.1


# GRID etc
dt = 0.001

h = 0.05
x_max = 5
y_max = 2

J = int(x_max / h)
h = x_max / J
K = int(y_max / h)

# set up boundary
N_theta = 100
L0 = 3
d_theta = L0 / N_theta

X = np.array([np.arange(0, L0, d_theta), np.zeros(N_theta)]).T + np.array([0.5, y_max/2])

Z0 = X[0, :].copy()
tether = {0:Z0}

# setup initial conditions
u0 = np.zeros((2, J, K))  # must be divergence-free
#u0[0, ...] = np.array([np.sin(np.pi * 2 * np.arange(K) / K) for j in range(J)])



if plotting:
    fig = plt.figure()
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()
    
t = 0

_x = h * np.array(range(J))
_y = h * np.array(range(K))
xy = np.meshgrid(_x, _y)

ct=0
while True:
    print(t)
    ct += 1
    freq = 2
    tether[0] = Z0 + 0.5 * np.array([0, np.sin(t * freq)])
    X, u2 = st.RK(X, u0, dt, h, K, J, d_theta, N_theta, _rho, _mu, tether)
    t += dt
    
    if plotting & (ct % 30 == 0):
        fig.clear()
        plt.pcolormesh(_x, _y, np.linalg.norm(u2, axis=0).T)    
        plt.quiver(xy[0].T[ ::3, ::3], xy[1].T[ ::3, ::3], u2[0, ::3, ::3], u2[1,  ::3, ::3], scale=0.4/h)
    #    plt.colorbar()
        plt.scatter(X[:, 0], X[:, 1] , s=2, color = 'black')
        plt.draw()
        plt.pause(0.02)
    u0 = np.copy(u2)
 