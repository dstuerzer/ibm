import numpy as np
import operators as op
import delta

def printu(u):
    print(u.T[::-1,:])
    
def xy(i, j):
    return (i % N_x) * h, (j % N_y) * h
    
# set up grid
h = 1
x_max = 4
y_max = 5

N_x = int(x_max / h)
h = x_max / N_x
N_y = int(y_max / h)
y_max = N_y * h

# set up boundary
N_theta = 20
d_theta = 2 * np.pi / N_theta


# setup initial conditions
u = np.zeros((2, N_x, N_y))  # must be divergence-free
u[1, 2, 3] = 1
u[0, 1, 1] = -1
phi = np.random.normal(0, 1, size=(N_x, N_y))

printu(op.L2(u, h)[0, ...])

print(xy(3,6))


from matplotlib import pyplot as plt

x = np.linspace(-3, 3, 100)

plt.plot(x, [delta.delta(xx) for xx in x])
plt.show()



