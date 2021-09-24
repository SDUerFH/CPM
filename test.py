import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    figure = plt.figure()
    ax = Axes3D(figure)
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    Z = np.exp(-D2 / 2.0 / sigma / sigma)
    print(Z)
    ax.plot_surface(gridx, gridy, Z, cmap='rainbow')
    plt.show()


guassian_kernel(45, 45, 40, 40, 21)
