import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import scipy.io as sio

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

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

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

def getCausalPatchesWPixel(I, dim):
    """
    Same as getCausalPatches, except include the center pixel
    """
    P = getPatches(I, dim)
    k = int((dim*dim-1)/2)+1
    P = P[:, :, 0:k]
    return P

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

def unwrapColorPatch(p, K):
    p = np.reshape(p, [3, K, K])
    return np.rollaxis(p, 0, 3)

def recombineColorPatches(I, dim, M, N):
    IFinal = np.zeros((M, N, 3))
    Weights = np.zeros((M, N))
    d = int((dim-1)/2)
    g = np.arange(dim)-d
    g = np.exp(-g**2/(2*(0.5*d)**2))
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            p = unwrapColorPatch(I[i, j, :], dim)
            p = (p*g[None, :, None])*g[:, None, None]
            IFinal[i:i+dim, j:j+dim, :] += p
            Weights[i:i+dim, j:j+dim] += g[None, :]*g[:, None]
    IFinal /= Weights[:, :, None]
    return IFinal

def testDictionary(NFactors, K, lam, showDict = True):
    import spams
    from VideoTools import loadImageIOVideo
    import time
    (I, IDims) = loadImageIOVideo("jumpingjacks2men.ogg")
    
    #Step 1: Build dictionary based on first frame
    F1 = np.reshape(I[0, :], IDims)
    P = getColorPatchesImageSet([F1], K, getPatches)
    ShapeBefore = P.shape[1::]
    P = np.reshape(P, [P.shape[0]*P.shape[1]*P.shape[2], P.shape[3]])
    print("Doing NNSC for %i factors on %i patches..."%(NFactors, P.shape[0]))
    tic = time.time()
    U = spams.nnsc(P.T, lambda1 = lam, return_lasso = False, K = NFactors, numThreads = 8)
    print("Elapsed Time: %g"%(time.time() - tic))
    if showDict:
        k = int(np.sqrt(NFactors))
        for i in range(k*k):
            plt.subplot(k, k, i+1)
            p = unwrapColorPatch(U[:, i], K)
            p = p/np.max(p)
            plt.imshow(p, interpolation = 'none')
            plt.axis('off')
        plt.savefig("Dict_%i_%i_%.3g.png"%(NFactors, K, lam), bbox_inches = 'tight')
    
    #Step 2: Represent another frame in the constructed dictionary
    F2 = np.reshape(I[10, :], IDims)
    P2 = getColorPatchesImageSet([F2], K, getPatches)
    ShapeBefore = P2.shape[1::]
    P2 = np.reshape(P2, [P2.shape[0]*P2.shape[1]*P2.shape[2], P2.shape[3]])    
    print("Solving for factors")
    tic = time.time()
    Alpha = spams.lasso(P2.T, U, lambda1 = lam, pos = True, numThreads = 8)
    print("Elapsed Time: %g"%(time.time() - tic))
    P2New = np.reshape((U*Alpha).T, ShapeBefore)
    F2Reconstructed = recombineColorPatches(P2New, K, F2.shape[0], F2.shape[1])
    writeImage(F2, "Orig.png")
    writeImage(F2Reconstructed, "Reconstructed_%i_%i_%.3g.png"%(NFactors, K, lam))
    print("%i Factors, Avg nonzero elems: %.3g"%(NFactors, NFactors*Alpha.size/float(Alpha.shape[0]*Alpha.shape[1])))
    #plt.hist(Alpha.flatten())
    #plt.title("$\\alpha$ Histogram")
    #plt.savefig("Hist_%i_%i_%.3g.png"%(NFactors, K, lam), bbox_inches = 'tight')

if __name__ == '__main__':
    K = 5
    for NFactors in [64, 121, 256]:
        testDictionary(NFactors, K, 1.0)