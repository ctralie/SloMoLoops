import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.misc
from skimage.transform import pyramid_gaussian
import pyflann
from PatchTools import *
from sklearn.decomposition import PCA
import time

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
        offi = int(dI[n]+i)
        offj = int(dJ[n]+j)
        imidx = int(BpLidx[offi, offj][0])
        ni = BpLidx[offi, offj][1]
        nj = BpLidx[offi, offj][1]
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

def getPatchDictionaries(As, Aps, NLevels = 3, KSpatials = [5, 5], patchfn = getColorPatchesImageSet, kPCA = 5):
    """
    :param As: An array of images of the same dimension
    :param Aps: An array of images of the same dimension as As, parallel to As
    """
    #Make image pyramids
    ALs = []
    ApLs = []
    for i in range(len(As)):
        ALs.append(tuple(pyramid_gaussian(As[i], NLevels, downscale = 2)))
        ApLs.append(tuple(pyramid_gaussian(Aps[i], NLevels, downscale = 2)))
    #Make ANN lists
    annLists = {}
    Xs = {}
    annidx2idxs = {}
    pcaBases = {}
    X = np.array([])
    for level in range(NLevels, -1, -1):
        pcaBases[level] = None
        X = np.array([])
        print("Level %i"%level)
        KSpatial = KSpatials[-1]
        if level == 0:
            KSpatial = KSpatials[0]
        #Step 1: Make features
        for i in range(len(ALs)):
            APatches = patchfn([ALs[i][level]], KSpatial, getPatches)
            ApPatches = patchfn([ApLs[i][level]], KSpatial, getCausalPatches)
            Xi = np.concatenate((APatches, ApPatches), 3)
            print("X.shape = {}".format(X.shape))
            if level < NLevels:
                #Use multiresolution features
                As2 = [scipy.misc.imresize(ALs[i][level+1], ALs[i][level].shape)]
                Aps2 = [scipy.misc.imresize(ApLs[i][level+1], ApLs[i][level].shape)]
                A2Patches = patchfn(As2, KSpatial, getPatches)
                Ap2Patches = patchfn(Aps2, KSpatial, getPatches)
                Xi = np.concatenate((Xi, A2Patches, Ap2Patches), 3)
            XiShape = Xi.shape
            Xi = np.reshape(Xi, [Xi.shape[0]*Xi.shape[1]*Xi.shape[2], Xi.shape[3]])
            if not pcaBases[level]:
                pca = PCA(n_components = kPCA)
                pca.fit(Xi)
                print("Explained Variance PCA Level %i: %g"%(level, \
                    np.sum(pca.explained_variance_ratio_)))
                pcaBases[level] = pca
            Xi = pcaBases[level].transform(Xi)
            Xi = np.reshape(Xi, [XiShape[0], XiShape[1], XiShape[2], Xi.shape[1]])
            if X.size == 0:
                X = Xi
            else:
                X = np.concatenate((X, Xi), 3)
        annList = pyflann.FLANN()
        annList.build_index(np.reshape(X, [X.shape[0]*X.shape[1]*X.shape[2], X.shape[3]]))
        annLists[level] = annList
        Xs[level] = X

    return {'annLists':annLists, 'ALs':ALs, 'ApLs':ApLs, 'Xs':Xs, 'pcaBases':pcaBases}

def doImageAnalogies(As, Aps, B, Kappa = 0.0, NLevels = 3, KSpatials = [5, 5], patchfn = getColorPatchesImageSet):
    """
    :param As: An array of images of the same dimension
    :param Aps: An array of images of the same dimension as As, parallel to As
    :param B: The example image
    """
    print("Getting patch dictionaries...")
    res = getPatchDictionaries(As, Aps, NLevels = NLevels, KSpatials = KSpatials, patchfn = patchfn)
    [ALs, ApLs, Xs, annLists, pcaBases] = [res['ALs'], res['ApLs'], res['Xs'], res['annLists'], res['pcaBases']]

    print("Doing image analogies...")
    BL = tuple(pyramid_gaussian(B, NLevels, downscale = 2))
    BpL = []
    BpLidx = []
    for i in range(len(BL)):
        BpL.append(np.zeros(BL[i].shape, dtype = np.int64))
        BpLidx.append(-1*np.ones((BL[i].shape[0], BL[i].shape[1], 3)))

    #Do multiresolution synthesis
    for level in range(NLevels, -1, -1):
        KSpatial = KSpatials[-1]
        if level == 0:
            KSpatial = KSpatials[0]
        #Step 1: Make features
        B2 = None
        Bp2 = None
        if level < NLevels:
            #Use multiresolution features
            B2 = scipy.misc.imresize(BL[level+1], BL[level].shape)
            Bp2 = scipy.misc.imresize(BpL[level+1], BpL[level].shape)

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
                #Put F into PCA basis
                F = F.flatten()
                F = pcaBases[level].transform(F[None, :])
                #Find index of most closely matching feature point in A
                idx = annLists[level].nn_index(F)[0].flatten()
                X = Xs[level]
                idx = np.unravel_index(idx, (X.shape[0], X.shape[1], X.shape[2]))
                idx = np.array(idx, dtype = np.int64).flatten()
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

def flattenPatchArray(X):
    X = np.reshape(X, [X.shape[0]*X.shape[1]*X.shape[2], X.shape[3]])
    return X

def doImageAnalogiesAcausal(As, Aps, Bs, KSpatial = 5, \
        patchfn = getColorPatchesImageSet, outputIters = False):
    BPatches = []
    minDists = []
    BpPatches = []
    ShapesBefore = []
    for B in Bs:
        BPatches.append(patchfn([B], KSpatial, getPatches))
        ShapesBefore.append(BPatches[-1].shape[1::])
        BPatches[-1] = flattenPatchArray(BPatches[-1])
        minDists.append(np.array([]))
        BpPatches.append(np.array([]))
    
    for i in range(len(As)):
        #Loop through for memory reasons
        APatches = patchfn([As[i]], KSpatial, getPatches)
        ApPatches = patchfn([Aps[i]], KSpatial, getPatches)
        APatches = flattenPatchArray(APatches)
        ApPatches = flattenPatchArray(ApPatches)
        #Build ANN function off of A patches
        annList = pyflann.FLANN()
        annList.build_index(APatches)
        
        tic = time.time()
        for j, B in enumerate(Bs):
            (idx, dists) = annList.nn_index(BPatches[j])
            if i == 0:
                BpPatches[j] = ApPatches[idx, :]
                minDists[j] = dists
            else:
                bidx = np.arange(BpPatches[j].shape[0])
                bidx = bidx[dists < minDists[j]]
                BpPatches[j][bidx] = ApPatches[idx[bidx], :]
                minDists[j][bidx] = dists[bidx]
                print("Frame %i %g%s Better Patches"%\
                    (j, 100*float(len(bidx))/BpPatches[j].shape[0], '%'))
            if outputIters:
                Bp = recombineColorPatches(np.reshape(BpPatches[j], ShapeBefore),
                                            KSpatial, B.shape[0], B.shape[1])
                writeImage(Bp, "BPIter%i_%i.png"%(j, i))
        print("Elapsed Time NN %i Frames iter %i of %i: %g"%\
                    (len(Bs), i, len(As), time.time() - tic))
    Bps = []
    for thisBpPatches, ShapeBefore in zip(BpPatches, ShapesBefore):
        thisBpPatches = np.reshape(thisBpPatches, ShapeBefore)
        Bps.append(recombineColorPatches(thisBpPatches, KSpatial, B.shape[0], B.shape[1]))
    return Bps

def testSuperRes(fac, Kappa, NLevels, fileprefix):
    from VideoTools import loadImageIOVideo
    import skimage.transform
    (I, _, IDims) = loadImageIOVideo("jumpingjacks2men.ogg")
    IDims2 = (int(IDims[0]*fac), int(IDims[1]*fac))
    As = []
    Aps = []
    for i in range(100, 120):
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
    #Bp = doImageAnalogies(As, Aps, B, Kappa = Kappa, NLevels = NLevels)
    Bp = doImageAnalogiesAcausal(As, Aps, B, KSpatial = 5, outputIters = True)
    writeImage(Bp, "%sBP.png"%fileprefix)

if __name__ == '__main__':
    testSuperRes(fac = 0.25, Kappa = 1, NLevels = 2, fileprefix = "")
