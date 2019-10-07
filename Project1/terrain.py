import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import random, seed

from Class import Regression, Error, Resample#, Error_Analysis, Resampling


from getData import *


#Code from project description
#load terrain
terrain1 = imread('SRTM_data_Norway_1.tif')

"""
#show terrain
plt.figure()
plt.imshow(terrain1, cmap='gray')
plt.title('Terrain over Stavanger, Norway')
plt.xlabel('X')
plt.ylabel('Y')
#plt.show()
"""

#for memory purposes we pick a smaller part of the terrain data to work with
terrain_fit = terrain1[0:250, 0:250]

"""
#show the chosen terrain
plt.figure()
plt.imshow(terrain_fit, cmap='gray')
plt.title('Snippet of terrain data')
plt.xlabel('X')
plt.ylabel('Y')
plt.colorbar(label ='Z')
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
"""


def CreateDesignMatrix(x, y, n=5):

    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    p = int((n + 1)*(n + 2)/2)
    X = np.ones((N,p))

    for i in range(1, n + 1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = x**(i-k) * y**k

    return X


n_x= 250       #number of points
n_y= 250
m=7        # degree of polynomial

# sort the random values, else your fit will go crazy
x = np.sort(np.random.uniform(0, 1, n_x))
y = np.sort(np.random.uniform(0, 1, n_y))


# use the meshgrid functionality, very useful
x, y = np.meshgrid(x,y)
#z = FrankeFunction(x, y)
z = np.ravel(terrain_fit)

#Transform from matrices to vectors
x_1=np.ravel(x)
y_1=np.ravel(y)
n=int(len(x_1))
z_1=np.ravel(z)+ np.random.random(n) * 1


#finally create the design matrix
X = CreateDesignMatrix(x_1,y_1,n=m)

getData_noRes('OLS', x_1,y_1,z_1,m, Print_errors=True, plot_err=True, plot_BiVar=False)
#def getData_Res_bootstrap(method,n_bootstraps, x,y,z,max_degree, Print_Val=True, plot_err=True, plot_BiVar=True):
#getData_Res_bootstrap('OLS', 100, x_1, y_1, z_1, m, Print_Val=True, plot_err=False, plot_BiVar=True)
