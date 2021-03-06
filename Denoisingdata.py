from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math

# Hi guys , I'm Mohammad Javad Rajabi, so let's dive in 

# utility function
def get_main_function():
    x = np.linspace(0, math.pi * 3 / 2, 30)
    y = np.linspace(0, math.pi * 3 / 2, 30)
    X, Y = np.meshgrid(x, y)
    return np.sin(X * Y)


def make_function_noisy(z):
    max_noise = 0.1
    t = 2 * np.random.rand(30 * 30) * max_noise - max_noise
    t = t.reshape(30, 30)
    return np.add(z, t)


def get_function():
    return make_function_noisy(get_main_function())


def show_my_matrix(Z):
    x = np.linspace(0, math.pi * 3 / 2, 30)
    y = np.linspace(0, math.pi * 3 / 2, 30)
    X, Y = np.meshgrid(x, y)
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='black')
    ax.set_title('surface')
    ax.view_init(60, 35)
    plt.show()


def show_main():
    Z = get_main_function()
    show_my_matrix(Z)


def show_noisy():
    Z = get_function()
    show_my_matrix(Z)


# forming SVD decomposition
z = get_main_function()
u , s ,vt = np.linalg.svd(z)

Sigma = np.zeros((u.shape[1], vt.shape[0]))
for i in range(u.shape[1]):
    Sigma[i, i] = s[i]
# np.fill_diagonal(Sigma, s)

reconstructed =  u @ Sigma @ vt


# denoising
k = 15 
approx_img = u @ Sigma[..., :k] @ vt[..., :k, :]


# showing
show_noisy()
show_main()
show_my_matrix(approx_img)