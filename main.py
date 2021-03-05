import numpy as np
import operators as op
import stepper as st
from matplotlib import pyplot as plt
import delta
import time

 # %matplotlib qt
       
def printu(u):
    return u.T[::-1, :]

# def xy(i, j):
 #   return (i % N_x) * h, (j % N_y) * h
    
# set up grid
plotting = False

h = 0.05
x_max = 5
y_max = 5

J = int(x_max / h)
h = x_max / J
K = int(y_max / h)

# set up boundary
N_theta = 100
d_theta = 2 * np.pi / N_theta
X = np.array([np.array([1.1*np.cos(s * d_theta) + J * h/2, np.sin(s * d_theta) + K * h /2]) for s in range(N_theta)])



# setup initial conditions
u0 = np.zeros((2, J, K))  # must be divergence-free
u0[0, ...] = np.array([np.sin(np.pi * 2 * np.arange(K) / K) for j in range(J)])


# parameters

_tension_K = 1
_mu = 0.01
_rho = 1
dt = 0.02
if plotting:
    fig = plt.figure()
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()
t = 0

_x = h * np.array(range(J))
_y = h * np.array(range(K))
xy = np.meshgrid(_x, _y)

for ct in range(500):
    print(ct)
    X, u2 = st.RK(X, u0, dt, h, K, J, _tension_K, d_theta, N_theta, _rho, _mu)
    t += dt
    
    
    
    if plotting:
        fig.clear()
        plt.pcolormesh(_x, _y, np.linalg.norm(u2, axis=0).T)    
        plt.quiver(xy[0].T[ ::3, ::3], xy[1].T[ ::3, ::3], u2[0, ::3, ::3], u2[1,  ::3, ::3], scale=1/h)
        #plt.colorbar()
        plt.scatter(X[:, 0], X[:, 1] , s=2, color = 'black')
        plt.draw()
        plt.pause(0.1)
    u0 = np.copy(u2)
   # time.sleep(3)
plt.show() 
    
                

