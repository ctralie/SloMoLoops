import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.misc
from skimage.transform import pyramid_gaussian

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def readImage(filename):
    I = scipy.misc.imread(filename)
    I = np.array(I, dtype=np.float32)/255.0
    return I

def writeImage(I, filename):
    IRet = I*255.0
    IRet[IRet > 255] = 255
    IRet[IRet < 0] = 0
    IRet = np.array(IRet, dtype=np.uint8)
    scipy.misc.imsave(filename, IRet)

def getPatches(I, dim):
    """
    Given an an MxN image I, get all dimxdim dimensional
    patches
    :return (M-dim+1)x(N-dim+1)x(dimxdim) array of patches
    """
    #http://stackoverflow.com/questions/13682604/slicing-a-numpy-image-array-into-blocks
    shape = np.array(I.shape*2)
    strides = np.array(I.strides*2)
    W = np.asarray(dim)
    shape[I.ndim:] = W
    shape[:I.ndim] -= W - 1
    if np.any(shape < 1):
        raise ValueError('Window size %i is too large for image'%dim)
    P = np.lib.stride_tricks.as_strided(I, shape=shape, strides=strides)
    P = np.reshape(P, [P.shape[0], P.shape[1], dim*dim])
    return P

def getCausalPatches(I, dim):
    """
    Assuming dim is odd, return L-shaped patches that would
    occur in raster order
    """
    P = getPatches(I, dim)
    k = int((dim*dim-1)/2)
    P = P[:, :, 0:k]
    return P

def getCoherenceMatch(X, x0, BpLidx, dim, i, j):
    """
    :param X: An MxNxDimFeatures array of feature vectors at each pixel
    :param x0: The feature vector of the pixel that's being filled in
    :param BpLidx: An MxN array of raveled indices from which pixels have
        been drawn so far
    :param dim: Dimension of patch
    :param i: Row of pixel
    :param j: Column of pixel
    """
    k = int((dim*dim-1)/2)
    M = X.shape[0]
    N = X.shape[1]
    minDistSqr = np.inf
    idxmin = [-1, -1]
    [dJ, dI] = np.meshgrid(np.arange(dim), np.arange(dim))
    dI = np.array(dI.flatten()[0:k], dtype = np.int64) - (dim-1)/2
    dJ = np.array(dJ.flatten()[0:k], dtype = np.int64) - (dim-1)/2
    #TODO: Vectorize code below
    for n in range(dI.size):
        #Indices of pixel picked for neighbor
        ni = BpLidx[dI[n]+i, dJ[n]+j][0]
        nj = BpLidx[dI[n]+i, dJ[n]+j][1]
        if ni == -1 or nj == -1:
            continue
        ni = int(ni - dI[n])
        nj = int(nj - dJ[n])
        if ni < 0 or nj < 0 or ni >= M or nj >= N:
            continue
        x = X[ni, nj, :]
        distSqr = np.sum((x - x0)**2)
        if distSqr < minDistSqr:
            minDistSqr = distSqr
            idxmin = [ni, nj]
    return (idxmin, minDistSqr)


def doImageAnalogies(A, Ap, B, Kappa = 0.0, NLevels = 3, KSpatials = [5, 5]):
    import pyflann
    #Make image pyramids
    AL = tuple(pyramid_gaussian(A, NLevels, downscale = 2))
    ApL = tuple(pyramid_gaussian(Ap, NLevels, downscale = 2))
    BL = tuple(pyramid_gaussian(B, NLevels, downscale = 2))
    BpL = []
    BpLidx = []
    print("BL:")
    for i in range(len(BL)):
        print(BL[i].shape)
        BpL.append(np.zeros(BL[i].shape))
        BpLidx.append(-1*np.ones((BL[i].shape[0], BL[i].shape[1], 2)))
    print("AL:")
    for i in range(len(AL)):
        print(AL[i].shape)

    #Do multiresolution synthesis
    for level in range(NLevels, -1, -1):
        KSpatial = KSpatials[-1]
        if level == 0:
            KSpatial = KSpatials[0]
        #Step 1: Make features
        APatches = getPatches(rgb2gray(AL[level]), KSpatial)
        ApPatches = getCausalPatches(rgb2gray(ApL[level]), KSpatial)
        X = np.concatenate((APatches, ApPatches), 2)
        print("X.shape = ", X.shape)
        B2 = None
        Bp2 = None
        if level < NLevels:
            #Use multiresolution features
            A2 = scipy.misc.imresize(AL[level+1], AL[level].shape)
            Ap2 = scipy.misc.imresize(ApL[level+1], ApL[level].shape)
            A2Patches = getPatches(rgb2gray(A2), KSpatial)
            Ap2Patches = getPatches(rgb2gray(Ap2), KSpatial)
            X = np.concatenate((X, A2Patches, Ap2Patches), 2)
            B2 = scipy.misc.imresize(BL[level+1], BL[level].shape)
            Bp2 = scipy.misc.imresize(BpL[level+1], BpL[level].shape)
        annList = pyflann.FLANN()
        annList.build_index(np.reshape(X, [X.shape[0]*X.shape[1], X.shape[2]]))

        #Step 2: Fill in the first few scanLines to prevent the image
        #from getting crap in the beginning
        if level == NLevels:
            I = np.array(ApL[level]*255, dtype = np.uint8)
            I = scipy.misc.imresize(I, BpL[level].shape)
            BpL[level] = np.array(I/255.0, dtype = np.float64)
        else:
            I = np.array(BpL[level+1]*255, dtype = np.uint8)
            I = scipy.misc.imresize(I, BpL[level].shape)
            BpL[level] = np.array(I/255.0, dtype = np.float64)

        #Step 3: Fill in the pixels in scanline order
        d = int((KSpatial-1)/2)
        for i in range(d, BpL[level].shape[0]-d):
            print(i)
            for j in range(d, BpL[level].shape[1]-d):
                #Make the feature at this pixel
                #Full patch B
                BPatch = rgb2gray(BL[level][i-d:i+d+1, j-d:j+d+1, :])
                #Causal patch B'
                BpPatch = rgb2gray(BpL[level][i-d:i+d+1, j-d:j+d+1, :]).flatten()
                BpPatch = BpPatch[0:int((KSpatial*KSpatial-1)/2)]
                F = np.concatenate((BPatch.flatten(), BpPatch.flatten()))

                if level < NLevels:
                    #Use multiresolution features
                    BPatch = rgb2gray(B2[i-d:i+d+1, j-d:j+d+1, :])
                    BpPatch = rgb2gray(Bp2[i-d:i+d+1, j-d:j+d+1, :])
                    F = np.concatenate((F, BPatch.flatten(), BpPatch.flatten()))
                #Find index of most closely matching feature point in A
                #DistSqrFn = XSqr + np.sum(F**2) - 2*X.dot(F)
                idx = annList.nn_index(F)[0].flatten()
                idx = np.unravel_index(idx, (X.shape[0], X.shape[1]))
                if Kappa > 0:
                #Compare with coherent pixel
                    (idxc, distSqrc) = getCoherenceMatch(X, F, BpLidx[level], KSpatial, i, j)
                    distSqr = np.sum((X[idx[0], idx[1]] - F)**2)
                    fac = 1 + Kappa*(2.0**(level - NLevels))
                    if distSqrc < distSqr*fac*fac:
                        idx = idxc
                BpLidx[level][i, j, :] = idx
                BpL[level][i, j, :] = ApL[level][idx[0]+d, idx[1]+d, :]
            if i%20 == 0:
                writeImage(BpL[level], "%i.png"%level)
        plt.subplot(122)
        plt.imshow(BpLidx[level][:, :, 0], cmap = 'Spectral')
        plt.title("Y")
        plt.subplot(121)
        plt.imshow(BpLidx[level][:, :, 1], cmap = 'Spectral')
        plt.title("X")
        plt.savefig("%i_idx.png"%level, bbox_inches = 'tight')
    return BpL[0]

def testCyclops():
    A = readImage("images/me-mask.png")
    Ap = readImage("images/me.jpg")
    B = readImage("images/cyclopsmask.png")
    res = doImageAnalogies(A, Ap, B, Kappa = 0.1, NLevels = 1)

def testSuperRes(fac, Kappa, NLevels, fileprefix):
    from SlidingWindowVideoTDA.VideoTools import loadImageIOVideo
    import skimage.transform
    (I, IDims) = loadImageIOVideo("jumpingjacks2men.ogg")
    Ap = np.reshape(I[0, :], IDims)
    IDims2 = (int(IDims[0]*fac), int(IDims[1]*fac))
    A = skimage.transform.resize(Ap, IDims2)
    A = skimage.transform.resize(A, IDims)
    BpGT = np.reshape(I[10, :], IDims)
    B = skimage.transform.resize(BpGT, IDims2)
    B = skimage.transform.resize(B, IDims)
    
    writeImage(A, "%sA.png"%fileprefix)
    writeImage(Ap, "%sAp.png"%fileprefix)
    writeImage(B, "%sB.png"%fileprefix)
    writeImage(BpGT, "%sBpGT.png"%fileprefix)
    Bp = doImageAnalogies(A, Ap, B, Kappa = Kappa, NLevels = NLevels)
    writeImage(Bp, "%sBP.png"%fileprefix)

if __name__ == '__main__':
    testSuperRes(fac = 0.25, Kappa = 0.1, NLevels = 2, fileprefix = "")
