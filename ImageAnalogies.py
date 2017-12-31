import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.misc
from skimage.transform import pyramid_gaussian
import spams
import time
from VideoTools import *
from PatchTools import *

def getPatchDictionaries(As, Aps, NLevels = 3, KSpatials = [5, 5], FactorsPerFrame = 256, lam = 1, patchfn = getColorPatchesImageSet, NSubsample = 100000):
    """
    :param As: An array of images of the same dimension
    :param Aps: An array of images of the same dimension as As, parallel to As
    """
    #Initialize dictionaries at each level
    Dicts = {}
    for level in range(NLevels, -1, -1):
        Dicts[level] = np.array([])
    for i in range(len(As)):
        #Make image pyramids
        AL = tuple(pyramid_gaussian(As[i], NLevels, downscale = 2))
        ApL = tuple(pyramid_gaussian(Aps[i], NLevels, downscale = 2))
        #Build dictionaries level by level
        for level in range(NLevels, -1, -1):
            print("Making dictionary frame %i of %i, level %i"%(i, len(As), level))
            KSpatial = KSpatials[-1]
            if level == 0:
                KSpatial = KSpatials[0]
            #Step 1: Make features
            X = patchfn([AL[level]], KSpatial, getPatches)
            ApPatches = patchfn([ApL[level]], KSpatial, getCausalPatchesWPixel)
            X = np.concatenate((X, ApPatches), 3)
            if level < NLevels:
                #Use multiresolution features
                As2 = [scipy.misc.imresize(AL[level+1], AL[level].shape)]
                Aps2 = [scipy.misc.imresize(ApL[level+1], ApL[level].shape)]
                A2Patches = patchfn(As2, KSpatial, getPatches)
                Ap2Patches = patchfn(Aps2, KSpatial, getPatches)
                X = np.concatenate((X, A2Patches, Ap2Patches), 3)
            Y = np.reshape(X, [X.shape[0]*X.shape[1]*X.shape[2], X.shape[3]])
            U = spams.nnsc(Y.T, lambda1 = lam, return_lasso = False, K = FactorsPerFrame)
            if i == 0:
                Dicts[level] = U
            else:
                #Take the union of all dictionaries for each image
                Dicts[level] = np.concatenate((Dicts[level], U), 1)
    return Dicts

def doImageAnalogies(As, Aps, B, Kappa = 0.0, NLevels = 3, KSpatials = [5, 5], \
                    FactorsPerFrame = 256, lam = 1.0, patchfn = getColorPatchesImageSet, NSubsample = 100000):
    """
    :param As: An array of images of the same dimension
    :param Aps: An array of images of the same dimension as As, parallel to As
    :param B: The example image
    """
    print("Getting patch dictionaries...")
    Dicts = getPatchDictionaries(As, Aps, NLevels = NLevels, KSpatials = KSpatials, \
                FactorsPerFrame = FactorsPerFrame, lam = lam, patchfn = patchfn, NSubsample = NSubsample)

    print("Doing image analogies...")
    ApL = tuple(pyramid_gaussian(Aps[0], NLevels, downscale = 2))
    BL = tuple(pyramid_gaussian(B, NLevels, downscale = 2))
    BpL = []
    for i in range(len(BL)):
        BpL.append(np.zeros(BL[i].shape))

    #Do multiresolution synthesis
    for level in range(NLevels, -1, -1):
        KSpatial = KSpatials[-1]
        if level == 0:
            KSpatial = KSpatials[0]
        
        #Select parts of dictionary corresponding to causal pixels
        #(assuming color patches)
        k = int((KSpatial*KSpatial-1)/2)+1
        idx = np.zeros(Dicts[level].shape[0])
        s = KSpatial*KSpatial*3
        for a in range(3):
            idx[s+k*(a+1)-1] = 1
        DCausal = np.asfortranarray(Dicts[level][idx == 0, :], dtype = np.float64)
        DColor = np.asfortranarray(Dicts[level][idx == 1, :], dtype = np.float64)

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
                BPatch = patchfn([BL[level][i-d:i+d+1, j-d:j+d+1, :]], KSpatial, getPatches)
                #Causal patch B'
                BpPatch = patchfn([BpL[level][i-d:i+d+1, j-d:j+d+1, :]], KSpatial, getCausalPatches)
                F = np.concatenate((BPatch.flatten(), BpPatch.flatten()))
                if level < NLevels:
                    #Use multiresolution features
                    BPatch = patchfn([B2[i-d:i+d+1, j-d:j+d+1, :]], KSpatial, getPatches)
                    BpPatch = patchfn([Bp2[i-d:i+d+1, j-d:j+d+1, :]], KSpatial, getPatches)
                    F = np.concatenate((F, BPatch.flatten(), BpPatch.flatten()))
                #Solve for F as a sparse linear combination of causal dictionary elements
                F = np.asfortranarray(F[:, None], dtype = np.float64)
                Alpha = spams.lasso(F, DCausal, lambda1 = lam, pos = True, numThreads = 8)
                Alpha = (Alpha.toarray()).flatten()
                BpL[level][i, j, :] = DColor.dot(Alpha)
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

def testSuperRes(fac, Kappa, NLevels, fileprefix):
    from VideoTools import loadImageIOVideo
    import skimage.transform
    (I, IDims) = loadImageIOVideo("jumpingjacks2men.ogg")
    IDims2 = (int(IDims[0]*fac), int(IDims[1]*fac))
    As = []
    Aps = []
    for i in [0]:#range(100, 120, 4):
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
    Bp = doImageAnalogies(As, Aps, B, Kappa = Kappa, NLevels = NLevels, NSubsample = -1)
    writeImage(Bp, "%sBP.png"%fileprefix)

if __name__ == '__main__':
    testSuperRes(fac = 0.25, Kappa = 0, NLevels = 2, fileprefix = "")