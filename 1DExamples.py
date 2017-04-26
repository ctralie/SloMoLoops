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

def getInterpolatedSignal(x, fac):
    N = len(x)
    idx = np.arange(N)
    idxx = np.linspace(0, N, fac*N+1)
    return interp.spline(idx, x, idxx)

def getReorderedConsensus(X, N, theta):
    M = X.shape[0]
    d = X.shape[1]
    tu = np.unwrap(theta)
    if tu[-1] - tu[0] < 0:
        tu = -tu
    tu = tu - np.min(tu)
    NPeriods = int(np.round(np.max(tu)/(2*np.pi)*N/M))
    print "NPeriods = ", NPeriods
    tu = np.mod(tu, 2*np.pi)
    idx = np.argsort(tu)
    X2 = X[idx, :]
    t1 = tu[idx]

    t2 = np.linspace(0, 2*np.pi, M)
    f = interp.interp2d(np.arange(X2.shape[1]), t1, X2, kind='linear')
    Y = f(np.arange(X2.shape[1]), t2)
    Z = np.nan*np.ones((d, N))
    for i in range(d):
        idx = np.arange(M) + i*NPeriods
        idx = np.mod(idx, N)
        Z[i, idx] = Y[:, i]
    plt.imshow(Z, aspect = 'auto', interpolation = 'none', cmap = 'afmhot')
    plt.show()
    z = np.nanmedian(Z, 0)
    return z

if __name__ == '__main__':
    np.random.seed(100)
    Weighted = True
    NPeriods = 20
    SamplesPerPeriod = 9
    N = NPeriods*SamplesPerPeriod
    t = np.linspace(0, 2*np.pi*NPeriods, N)#N+5)[0:N]
    t2 = np.linspace(0, 2*np.pi, N)
    x = np.cos(t) + np.cos(3*t) + np.cos(4*t)
    x = x + 0.3*np.random.randn(len(x))
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

    #Do denoising
    y = getReorderedConsensus(X, len(x), theta)

    #Make color array
    c = plt.get_cmap('Spectral')
    C = c(np.array(np.round(np.linspace(0, 255, X.shape[0])), dtype=np.int32))
    C = C[:, 0:3]

    #Perform PCA down to 2D for visualization
    pca = PCA(n_components = 2)
    Y = pca.fit_transform(X)
    eigs = pca.explained_variance_


    fig = plt.figure(figsize=(18, 12))
    plt.subplot(241)
    drawLineColored(t, x, C[xidx, :])
    ax = plt.gca()
    plotbgcolor = (0.15, 0.15, 0.15)
    ax.set_axis_bgcolor(plotbgcolor)
    plt.ylim([-3, 3])
    plt.title("Original Signal")
    plt.xlabel("t")

    #ax2 = fig.add_subplot(132, projection = '3d')
    plt.subplot(242)
    plt.title("Sliding Window Adjacency Matrix\nWin = %i, $\kappa = %g$"%(dim, kappa))
    plt.imshow(1-A, cmap='gray')


    plt.subplot(243)
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

    plt.subplot(245)
    plt.scatter(v[:, 0], v[:, 1], 20, c=C, edgecolor = 'none')
    plt.xlabel("Laplacian Eigenvector 1")
    plt.ylabel("Laplacian Eigenvector 2")
    plt.title("Laplacian Eigenmap")
    ax = plt.gca()
    ax.set_axis_bgcolor(plotbgcolor)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.subplot(246)
    plt.plot(theta)

    plt.subplot(247)
    plt.plot(x2)
    plt.ylim([-3, 3])


    plt.subplot(244)
    plt.plot(y)
    plt.show()
