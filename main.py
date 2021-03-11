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

_mu = 0.003
_rho = 0.1


# GRID etc
dt = 0.0001

h = 0.05
x_max = 5
y_max = 4

J = int(x_max / h)
h = x_max / J
K = int(y_max / h)

# set up boundary
Z0 = np.array([0.5, 1.4])
Z1 = np.array([1.7, 2.4])

N_theta = 50
tick = np.linspace(0,1, N_theta)

L = np.linalg.norm(Z0 - Z1)
d_theta = L / (N_theta + 1)

X = np.array([Z0 + q*(Z1 - Z0) for q in tick])


tether = {0:Z0, N_theta-1:Z1}

# setup initial conditions
u0 = np.zeros((2, J, K))  # must be divergence-free
#u0[0, ...] = np.array([np.sin(np.pi * np.arange(K) / K) for j in range(J)])



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
    X, u2 = st.RK(X, u0, dt, h, K, J, d_theta, N_theta, _rho, _mu, tether)
    t += dt
    
    if plotting & (ct % 50 == 0):
        fig.clear()
        plt.pcolormesh(_x, _y, np.linalg.norm(u2, axis=0).T)    
        d_quiver = max(1, max(J//20, K//20))
        plt.quiver(xy[0].T[ ::d_quiver, ::d_quiver],
                   xy[1].T[ ::d_quiver, ::d_quiver],
                   u2[0, ::d_quiver, ::d_quiver],
                   u2[1, ::d_quiver, ::d_quiver])#, scale=0.4/h)
    #    plt.colorbar()
        plt.scatter(X[:, 0], X[:, 1] , s=2, color = 'black')
        plt.draw()
        plt.pause(0.02)
    u0 = np.copy(u2)
 