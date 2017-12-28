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
    :param X: An NImagesxMxNxDimFeatures array of feature vectors at each pixel\
                in each image
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
        imidx = int(BpLidx[dI[n]+i, dJ[n]+j][0])
        ni = BpLidx[dI[n]+i, dJ[n]+j][1]
        nj = BpLidx[dI[n]+i, dJ[n]+j][1]
        if ni == -1 or nj == -1:
            continue
        ni = int(ni - dI[n])
        nj = int(nj - dJ[n])
        if ni < 0 or nj < 0 or ni >= M or nj >= N:
            continue
        x = X[imidx, ni, nj, :]
        distSqr = np.sum((x - x0)**2)
        if distSqr < minDistSqr:
            minDistSqr = distSqr
            idxmin = [imidx, ni, nj]
    return (idxmin, minDistSqr)

def getGrayscalePatchesImageSet(As, KSpatial, patchfn):
    if len(As) == 0:
        return None
    AllP = patchfn(rgb2gray(As[0]), KSpatial)[None, :, :, :]
    for A in As[1::]:
        P = patchfn(rgb2gray(A), KSpatial)[None, :, :, :]
        AllP = np.concatenate((AllP, P), 0)
    return AllP

def getColorPatchesImageSet(As, KSpatial, patchfn):
    if len(As) == 0:
        return None
    AllP = np.array([])
    for A in As:
        P = np.array([])
        for k in range(3):
            Pk = patchfn(A[:, :, k], KSpatial)
            if P.size == 0:
                P = Pk
            else:
                P = np.concatenate((P, Pk), 2)
        if AllP.size == 0:
            AllP = P[None, :, :, :]
        else:
            AllP = np.concatenate((AllP, P[None, :, :, :]), 0)
    return AllP

def doImageAnalogies(As, Aps, B, Kappa = 0.0, NLevels = 3, KSpatials = [5, 5], bruteForce = False, patchfn = getColorPatchesImageSet):
    """
    :param As: An array of images of the same dimension
    :param Aps: An array of images of the same dimension as As, parallel to As
    :param B: The example image
    """
    import pyflann
    #patchfn = getGrayscalePatchesImageSet
    patchfn = getColorPatchesImageSet
    #Make image pyramids
    ALs = []
    ApLs = []
    for i in range(len(As)):
        ALs.append(tuple(pyramid_gaussian(As[i], NLevels, downscale = 2)))
        ApLs.append(tuple(pyramid_gaussian(Aps[i], NLevels, downscale = 2)))
    BL = tuple(pyramid_gaussian(B, NLevels, downscale = 2))
    BpL = []
    BpLidx = []
    print("BL:")
    for i in range(len(BL)):
        print(BL[i].shape)
        BpL.append(np.zeros(BL[i].shape))
        BpLidx.append(-1*np.ones((BL[i].shape[0], BL[i].shape[1], 3)))
    print("AL:")
    for i in range(len(ALs[0])):
        print(ALs[0][i].shape)

    #Do multiresolution synthesis
    for level in range(NLevels, -1, -1):
        KSpatial = KSpatials[-1]
        if level == 0:
            KSpatial = KSpatials[0]
        #Step 1: Make features
        APatches = patchfn([ALs[i][level] for i in range(len(ALs))], KSpatial, getPatches)
        ApPatches = patchfn([ApLs[i][level] for i in range(len(ApLs))], KSpatial, getCausalPatches)
        X = np.concatenate((APatches, ApPatches), 3)
        print("X.shape = ", X.shape)
        B2 = None
        Bp2 = None
        if level < NLevels:
            #Use multiresolution features
            As2 = [scipy.misc.imresize(ALs[i][level+1], ALs[i][level].shape) for i in range(len(ALs))]
            Aps2 = [scipy.misc.imresize(ApLs[i][level+1], ApLs[i][level].shape) for i in range(len(ApLs))]
            A2Patches = patchfn(As2, KSpatial, getPatches)
            Ap2Patches = patchfn(Aps2, KSpatial, getPatches)
            X = np.concatenate((X, A2Patches, Ap2Patches), 3)
            B2 = scipy.misc.imresize(BL[level+1], BL[level].shape)
            Bp2 = scipy.misc.imresize(BpL[level+1], BpL[level].shape)
        if bruteForce:
            XSqr = np.sum(X**2, 3).flatten()
        else:
            annList = pyflann.FLANN()
            annList.build_index(np.reshape(X, [X.shape[0]*X.shape[1]*X.shape[2], X.shape[3]]))

        #Step 2: Fill in the first few scanLines to prevent the image
        #from getting crap in the beginning
        if level == NLevels:
            I = np.array(ApLs[0][level]*255, dtype = np.uint8)
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
                BPatch = patchfn([BL[level][i-d:i+d+1, j-d:j+d+1, :]], KSpatial, getPatches)
                #Causal patch B'
                BpPatch = patchfn([BpL[level][i-d:i+d+1, j-d:j+d+1, :]], KSpatial, getCausalPatches)
                F = np.concatenate((BPatch.flatten(), BpPatch.flatten()))

                if level < NLevels:
                    #Use multiresolution features
                    BPatch = patchfn([B2[i-d:i+d+1, j-d:j+d+1, :]], KSpatial, getPatches)
                    BpPatch = patchfn([Bp2[i-d:i+d+1, j-d:j+d+1, :]], KSpatial, getPatches)
                    F = np.concatenate((F, BPatch.flatten(), BpPatch.flatten()))
                #Find index of most closely matching feature point in A
                if bruteForce:
                    DistSqrFn = XSqr + np.sum(F**2) - 2*(X.dot(F)).flatten()
                    idx = np.argmin(DistSqrFn)
                else:
                    idx = annList.nn_index(F)[0].flatten()
                idx = np.unravel_index(idx, (X.shape[0], X.shape[1], X.shape[2]))
                if Kappa > 0:
                #Compare with coherent pixel
                    (idxc, distSqrc) = getCoherenceMatch(X, F, BpLidx[level], KSpatial, i, j)
                    distSqr = np.sum((X[idx[0], idx[1], idx[2], :] - F)**2)
                    fac = 1 + Kappa*(2.0**(level - NLevels))
                    if distSqrc < distSqr*fac*fac:
                        idx = idxc
                BpLidx[level][i, j, :] = idx
                BpL[level][i, j, :] = ApLs[idx[0]][level][idx[1]+d, idx[2]+d, :]
            if i%20 == 0:
                writeImage(BpL[level], "%i.png"%level)
        plt.subplot(131)
        plt.imshow(BpLidx[level][:, :, 0], cmap = 'Spectral')
        plt.title("idx")
        plt.subplot(133)
        plt.imshow(BpLidx[level][:, :, 1], cmap = 'Spectral')
        plt.title("Y")
        plt.subplot(132)
        plt.imshow(BpLidx[level][:, :, 2], cmap = 'Spectral')
        plt.title("X")
        plt.savefig("%i_idx.png"%level, bbox_inches = 'tight')
    return BpL[0]

def testCyclops():
    A = readImage("images/me-mask.png")
    Ap = readImage("images/me.jpg")
    B = readImage("images/cyclopsmask.png")
    res = doImageAnalogies(A, Ap, B, Kappa = 0.1, NLevels = 1)

def testSuperRes(fac, Kappa, NLevels, fileprefix, bruteForce = False):
    from SlidingWindowVideoTDA.VideoTools import loadImageIOVideo
    import skimage.transform
    (I, IDims) = loadImageIOVideo("jumpingjacks2men.ogg")
    IDims2 = (int(IDims[0]*fac), int(IDims[1]*fac))
    As = []
    Aps = []
    for i in range(100, 120, 3):
        Ap = np.reshape(I[i, :], IDims)
        A = skimage.transform.resize(Ap, IDims2)
        A = skimage.transform.resize(A, IDims)
        As.append(A)
        Aps.append(Ap)

    BpGT = np.reshape(I[10, :], IDims)
    B = skimage.transform.resize(BpGT, IDims2)
    B = skimage.transform.resize(B, IDims)
    
    writeImage(A, "%sA.png"%fileprefix)
    writeImage(Ap, "%sAp.png"%fileprefix)
    writeImage(B, "%sB.png"%fileprefix)
    writeImage(BpGT, "%sBpGT.png"%fileprefix)
    Bp = doImageAnalogies(As, Aps, B, Kappa = Kappa, NLevels = NLevels, bruteForce = bruteForce)
    writeImage(Bp, "%sBP.png"%fileprefix)

if __name__ == '__main__':
    testSuperRes(fac = 0.25, Kappa = 0.5, NLevels = 2, fileprefix = "", bruteForce = False)
