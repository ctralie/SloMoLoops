import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.interpolate as interp
from VideoReordering import *
from sklearn.decomposition import PCA
from ripser import ripser

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
    T = N/NPeriods #Period
    print "NPeriods = ", NPeriods
    print "N/NPeriods = ", float(N)/NPeriods
    tu = N*np.mod(tu, 2*np.pi)/(2*np.pi)
    idx = np.argsort(tu)
    X2 = X[idx, :]
    t1 = tu[idx]
    Z = np.nan*np.ones((M, N))
    for i in range(M):
        ts = t1[i] + NPeriods*np.arange(d)
        imin = int(np.ceil(np.min(ts)))
        imax = int(np.floor(np.max(ts)))
        t2 = np.arange(imin, imax+1)
        x = interp.spline(ts, X2[i, :], t2)
        Z[i, np.mod(t2, N)] = x
    plt.imshow(Z, aspect = 'auto', interpolation = 'none', cmap = 'afmhot')
    plt.xlabel("Time")
    plt.ylabel("Spline Interpolated Windows")
    plt.show()
    z = np.nanmedian(Z, 0)
    return z

if __name__ == '__main__':
    np.random.seed(100)
    Weighted = False #Whether to use the weighted graph laplacian
    NPeriods = 20
    SamplesPerPeriod = 9
    N = NPeriods*SamplesPerPeriod
    t = np.linspace(0, 2*np.pi*NPeriods, N)#N+5)[0:N]
    t2 = np.linspace(0, 2*np.pi, N)
    x = np.cos(t) + np.cos(3*t) + np.cos(4*t)
    x = x + 0.5*np.random.randn(len(x))
    x2 = np.cos(t2) + np.cos(3*t2) + np.cos(4*t2)

    dim = 20
    Tau = 1
    dT = 1
    (X, xidx) = getSlidingWindow(x, dim, Tau, dT)
    #X = X - np.mean(X, 1)[:, None]
    #X = X/np.sqrt(np.sum(X**2, 1))[:, None]
    extent = Tau*dim
    
    #Use rips filtration to guide Laplacian
    D = getSSM(X)
    Is = ripser.doRipsFiltrationDM(D, 1, coeff=41)
    I = Is[1]
    thresh = np.argmax(I[:, 1] - I[:, 0])
    thresh = I[thresh, 0]
    if Weighted:
        A = np.exp(-D**2/(2*thresh**2))
    else:
        A = np.zeros(D.shape)
        A[D <= thresh] = 1
        np.fill_diagonal(A, 0)
    (w, v) = getLaplacianEigsDense(A, 3)
    
    #Compute reordering based on Laplacian

    
    v = v[:, [1, 2]]
    theta = np.arctan2(v[:, 1], v[:, 0])
    
    ##Ground truth
    #theta = np.mod(t[0:len(theta)], 2*np.pi)
    ##
    
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
    ylims = [-2, 3.5]
    plt.subplot(241)
    drawLineColored(t, x, C[xidx, :])
    ax = plt.gca()
    plotbgcolor = (0.15, 0.15, 0.15)
    ax.set_axis_bgcolor(plotbgcolor)
    plt.ylim(ylims)
    plt.title("Original Signal")
    plt.xlabel("t")

    #ax2 = fig.add_subplot(132, projection = '3d')
    plt.subplot(242)
    plt.title("Sliding Window Adjacency Matrix\nWin = %i, thresh = %g"%(dim, thresh))
    plt.imshow(1-A, cmap='gray')


    plt.subplot(243)
    plt.scatter(v[:, 0], v[:, 1], 20, c=C, edgecolor = 'none')
    plt.xlabel("Laplacian Eigenvector 1")
    plt.ylabel("Laplacian Eigenvector 2")
    plt.title("Laplacian Eigenmap")
    ax = plt.gca()
    ax.set_axis_bgcolor(plotbgcolor)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.subplot(244)
    plt.plot(theta)
    plt.title("Circular Coordinates")
    
    plt.subplot(245)
    ripser.plotDGM(I)
    plt.title("H1")

    plt.subplot(246)
    plt.plot(x2)
    plt.ylim(ylims)
    plt.title("Ground Truth")

    plt.subplot(247)
    #drawLineColored(np.arange(len(xresort)), xresort, C[ridx])
    t = np.unwrap(theta)
    #plt.scatter(t, xresort, c = C[ridx, :], edgecolor = 'none')
    plt.plot(xresort)
    ax = plt.gca()
    plt.ylim(ylims)
    plt.title("Reordered Signal")

    plt.subplot(248)
    plt.plot(y)
    plt.ylim(ylims)
    plt.title("Spline Consensus reordering")
    plt.show()
