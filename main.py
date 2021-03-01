import numpy as np
import operators as op
import stepper as st
from matplotlib import pyplot as plt
import delta
import time

 # %matplotlib qt

   
# def xy(i, j):
 #   return (i % N_x) * h, (j % N_y) * h
    
# set up grid
h = 0.01
x_max = 4
y_max = 4

J = int(x_max / h)
h = x_max / J
K = int(y_max / h)

# set up boundary
N_theta = 100
d_theta = 2 * np.pi / N_theta
X = np.array([np.array([np.cos(s * d_theta) + J * h/2, np.sin(s * d_theta) + K * h /2]) for s in range(N_theta)])



# setup initial conditions
u0 = np.zeros((2, J, K))  # must be divergence-free
u0[0, ...] = np.array([np.sin(np.pi * 2 * np.arange(K) / K) for j in range(J)])

# parameters

_tension_K = 1
_mu = 0.1
_rho = 1
dt = 0.01
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1])
plt.show()
t = 0
for _ in range(2):
    X, u2 = st.RK(X, u0, dt, h, K, J, _tension_K, d_theta, N_theta, _rho, _mu)
    t += dt
    fig.clear()
   # plt.imshow(np.linalg.norm(u, axis=0))
    #plt.colorbar()
    plt.scatter(X[:, 0], X[:, 1])
    plt.draw()
    plt.pause(0.1)
    u0 = np.copy(u2)
plt.show() 
    
                


                