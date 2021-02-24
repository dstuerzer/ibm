import numpy as np

def D_hat(K, J, h):
    mesh = np.meshgrid(range(J), range(K))
    return np.array([1j/h * np.sin(2 * np.pi * mesh[0].T / J), 1j/h * np.sin(2 * np.pi * mesh[1].T / K)])

def L_hat(K, J, h):
    mesh = np.meshgrid(range(J), range(K))
    return -4/(h*h) * (np.square(np.sin(np.pi * mesh[0].T / J)) + np.square(np.sin(np.pi * mesh[1].T / K)))

print(L_hat(2,4,1))