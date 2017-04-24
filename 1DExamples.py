import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.interpolate as interp
from VideoReordering import *
from sklearn.decomposition import PCA

def drawLineColored(idx, x, C):
    plt.hold(True)
    for i in range(len(x)-1):
        plt.plot(idx[i:i+2], x[i:i+2], c=C[i, :])

def getSlidingWindow(x, dim, Tau, dT):
    N = len(x)
    NWindows = int(np.floor((N-dim*Tau)/dT))
    X = np.zeros((NWindows, dim))
    idx = np.arange(len(x))
    xidx = [] #Index the original samples into the sliding window array (assumes dT evenly divides 1)
    for i in range(NWindows):
        if dT*i == int(dT*i):
            xidx.append(i)
        idxx = dT*i + Tau*np.arange(dim)
        start = int(np.floor(idxx[0]))
        end = int(np.ceil(idxx[-1]))
        X[i, :] = interp.spline(idx[start:end+1], x[start:end+1], idxx)
    while len(xidx) < len(x):
        xidx.append(xidx[-1])
    return (X, xidx)

if __name__ == '__main__':
    Weighted = False
    NPeriods = 20
    SamplesPerPeriod = 9
    N = NPeriods*SamplesPerPeriod
    t = np.linspace(0, 2*np.pi*NPeriods, N)#N+5)[0:N]
    t2 = np.linspace(0, 2*np.pi, N)
    x = np.cos(t) + np.cos(3*t) + np.cos(4*t)
    x2 = np.cos(t2) + np.cos(3*t2) + np.cos(4*t2)
    
    dim = SamplesPerPeriod
    Tau = 1
    dT = 1
    (X, xidx) = getSlidingWindow(x, dim, Tau, dT)
    X = X - np.mean(X, 1)[:, None]
    X = X/np.sqrt(np.sum(X**2, 1))[:, None]
    sio.savemat("X.mat", {"X":X})
    extent = Tau*dim

    #Compute reordering based on Laplacian
    kappa = 0.1
    D = getSSM(X)
    if Weighted:
        A = getW(D, kappa)
    else:
        A = CSMToBinaryMutual(D, kappa)
    (w, v) = getLaplacianEigsDense(A, 4)
    v = v[:, [1, 2]]
    theta = np.arctan2(v[:, 1], v[:, 0])
    ridx = np.argsort(theta)
    xresort = x[ridx]

    #Make color array
    c = plt.get_cmap('Spectral')
    C = c(np.array(np.round(np.linspace(0, 255, X.shape[0])), dtype=np.int32))
    C = C[:, 0:3]

    #Perform PCA down to 2D for visualization
    pca = PCA(n_components = 2)
    Y = pca.fit_transform(X)
    eigs = pca.explained_variance_


    fig = plt.figure(figsize=(18, 12))
    plt.subplot(231)
    drawLineColored(t, x, C[xidx, :])
    ax = plt.gca()
    plotbgcolor = (0.15, 0.15, 0.15)
    ax.set_axis_bgcolor(plotbgcolor)
    plt.ylim([-3, 3])
    plt.title("Original Signal")
    plt.xlabel("t")

    #ax2 = fig.add_subplot(132, projection = '3d')
    plt.subplot(232)
    plt.title("Sliding Window Adjacency Matrix\nWin = %i, $\kappa = %g$"%(dim, kappa))
    plt.imshow(1-A, cmap='gray')


    plt.subplot(233)
    #drawLineColored(np.arange(len(xresort)), xresort, C[ridx])
    t = np.unwrap(theta)
    plt.scatter(t, xresort, c = C[ridx, :], edgecolor = 'none')
    ax = plt.gca()
    plotbgcolor = (0.15, 0.15, 0.15)
    ax.set_axis_bgcolor(plotbgcolor)
    plt.ylim([-3, 3])
    plt.xlim([np.min(t), np.max(t)])
    plt.xlabel("t")
    plt.title("Reordered Signal")

    plt.subplot(234)
    plt.scatter(v[:, 0], v[:, 1], 20, c=C, edgecolor = 'none')
    plt.xlabel("Laplacian Eigenvector 1")
    plt.ylabel("Laplacian Eigenvector 2")
    plt.title("Laplacian Eigenmap")
    ax = plt.gca()
    ax.set_axis_bgcolor(plotbgcolor)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.subplot(235)
    plt.plot(theta)
    
    plt.subplot(236)
    plt.plot(x2)
    plt.ylim([-3, 3])
    
    plt.show()
