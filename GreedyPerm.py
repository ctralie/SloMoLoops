import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from CSMSSMTools import getCSM

def getGreedyPerm(X, M, Verbose = False):
    """
    Purpose: Naive O(NM) algorithm to do the greedy permutation
    :param X: Nxd array of Euclidean points
    :param M: Number of points in returned permutation
    :returns: permutation (N-length array of indices)
    """
    #By default, takes the first point in the list to be the
    #first point in the permutation, but could be random
    perm = np.zeros(M, dtype=np.int64)
    lambdas = np.zeros(M)
    ds = getCSM(X[0, :][None, :], X).flatten()
    for i in range(1, M):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, getCSM(X[idx, :][None, :], X).flatten())
        if Verbose:
            interval = int(0.05*M)
            if interval > 0:
                if i%interval == 0:
                    print("Greedy perm %i%s done..."%(int(100.0*i/float(M)), "%"))
    return perm

def getBatchGreedyPerm(X, M, NBatches, Verbose = False):
    idxperm = np.random.permutation(X.shape[0])
    NPerChunk = int(float(X.shape[0])/NBatches)
    MPerChunk = int(1.0*M/float(NBatches))
    if Verbose:
        print("NPerChunk = %i, MPerChunk = %i"%(NPerChunk, MPerChunk))
    idxs = []
    for i in range(NBatches):
        if Verbose:
            print("Doing batch %i of %i..."%(i, NBatches))
        idxrange = idxperm[i*NPerChunk:(i+1)*NPerChunk]
        idx = getGreedyPerm(X[idxrange, :], MPerChunk, False)
        idxs += idxrange[np.array(idx)].tolist()
    return idxs


def testNoisyCircle():
    t = np.linspace(0, 10*np.pi, 30001)[0:30000]
    X = np.zeros((len(t), 2))
    X[:, 0] = np.cos(t)
    X[:, 1] = np.sin(t)
    np.random.seed(420)
    X = X + 0.1*np.random.randn(X.shape[0], 2)
    Y = getGreedyPerm(X, 1000, Verbose = True)['Y']
    
    plt.scatter(X[:, 0], X[:, 1], 40, edgecolor = 'none')
    plt.plot(Y[:, 0], Y[:, 1], 'rx')
    plt.show()

def testUnevenClusters():
    np.random.seed(101)
    X1 = np.random.randn(100, 2)
    X2 = np.random.randn(5000, 2)
    X2 += np.array([[10, 2]])
    X = np.concatenate((X1, X2), 0)

    idxs = getBatchGreedyPerm(X, 50, 10, True)
    Y = X[idxs, :]
    #print("Y.shape = {}".format(Y.shape))
    #Y = X[np.random.permutation(np.arange(X.shape[0]))[0:500], :]

    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1])
    plt.axis('equal')
    plt.subplot(122)
    plt.scatter(Y[:, 0], Y[:, 1])
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    #testNoisyCircle()
    testUnevenClusters()