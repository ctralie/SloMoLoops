import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.interpolate as interp
from VideoReordering import *
from Laplacian import *
from CSMSSMTools import *
from FundamentalFreq import *
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

def ReorderingExample1D():
    np.random.seed(100)
    useGroundTruth = False #Whether to use ground truth circular coordinates in the reordering
    Weighted = False #Whether to use the weighted graph laplacian
    NPeriods = 20
    SamplesPerPeriod = 12
    N = NPeriods*SamplesPerPeriod
    t = np.linspace(0, 2*np.pi*NPeriods, N)
    t2 = np.linspace(0, 2*np.pi, N)
    cs = [1.0, 1.0, 1.0]
    x = cs[0]*np.cos(t) + cs[1]*np.cos(3*t) + cs[2]*np.cos(5*t)
    x = x + 0.6*np.random.randn(len(x))
    x2 = cs[0]*np.cos(t2) + cs[1]*np.cos(3*t2) + cs[2]*np.cos(5*t2)

    doPlot = False
    dim = int(np.round(estimateFundamentalFreq(x, shortBias = 0.0, doPlot = doPlot)[0]))
    if doPlot:
        plt.show()
    dim = 2*dim
    print("dim = %i"%dim)
    (X, xidx) = getSlidingWindow(x, dim, 1, 1)

    #Use rips filtration to guide Laplacian
    D = getSSM(X)
    Is = ripser.doRipsFiltrationDM(D, 1, coeff=41)
    I = Is[1]
    thresh = np.argmax(I[:, 1] - I[:, 0])
    thresh = np.mean(I[thresh, :])
    if Weighted:
        res = getLapCircularCoordinatesSigma(D, thresh, doPlot = True)
    else:
        res = getLapCircularCoordinatesThresh(D, thresh)
    [w, v, theta, A] = [res['w'], res['v'], res['theta'], res['A']]

    ##Ground truth
    if useGroundTruth:
        theta = np.mod(t[0:len(theta)], 2*np.pi)

    tu = np.unwrap(theta)
    if tu[-1] - tu[0] < 0:
        tu = -tu
    tu = tu - np.min(tu)
    ridx = np.argsort(np.mod(tu, 2*np.pi))
    xresort = x[ridx]

    #Do denoising
    y = getReorderedConsensus1D(X, len(x), theta, doPlot = doPlot)

    #Make color array
    c = plt.get_cmap('Spectral')
    C = c(np.array(np.round(np.linspace(0, 255, X.shape[0])), dtype=np.int32))
    C = C[:, 0:3]

    #Perform PCA down to 2D for visualization
    pca = PCA(n_components = 2)
    Y = pca.fit_transform(X)
    eigs = pca.explained_variance_


    fig = plt.figure(figsize=(18, 12))
    ylims = [-3, 3.5]
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
    plt.imshow(A, cmap='gray')


    plt.subplot(243)
    plt.scatter(v[:, 1], v[:, 2], 20, c=C, edgecolor = 'none')
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
    plt.savefig("Paper/Figures/1DExample.svg", bbox_inches = 'tight')

if __name__ == '__main__':
    ReorderingExample1D()
