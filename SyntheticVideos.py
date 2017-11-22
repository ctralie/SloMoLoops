from SlidingWindowVideoTDA.VideoTools import *
from CSMSSMTools import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import time

def getCircleRotatingVideo():
    T = 15
    NPeriods = 20
    N = T*NPeriods
    t = np.linspace(0, 2*np.pi*NPeriods, N)
    idx = np.arange(200) - 100
    [X, Y] = np.meshgrid(idx, idx)
    cx = 50*np.cos(t)
    cy = 50*np.sin(t)
    I = np.zeros((N, X.size))
    for i in range(N):
        F = ((X-cx[i])**2 + (Y-cy[i])**2) < (10**2)
        I[i, :] = F.flatten()
    return (I, X.shape)
