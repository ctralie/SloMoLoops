"""
Programmer: Chris Tralie
Purpose: Some tools that load/save videos in Python.  Also tools for blurring
and corrupting videos by byte errors
"""
import numpy as np
import numpy.linalg as linalg
import time
import os
import subprocess
import matplotlib.image as mpimage
import scipy.misc
import scipy.signal
import sys
from scipy.ndimage import gaussian_gradient_magnitude
from PIL import Image

#Need these for saving 3D video

AVCONV_BIN = 'avconv'
TEMP_STR = "pymeshtempprefix"

#############################################################
####                  VIDEO I/O TOOLS                   #####
#############################################################

#Methods for converting to YCbCr (copied matrices from Matlab)
toNTSC = np.array([[0.2989, 0.5959, 0.2115], [0.587, -0.2744, -0.5229], [0.114, -0.3216, 0.3114]])
fromNTSC = np.linalg.inv(toNTSC)

def rgb2ntsc(F):
    return F.dot(toNTSC.T)

def ntsc2rgb(F):
    return F.dot(fromNTSC.T)

def rgb2gray(F, repDims = True):
    G = np.dot(F[...,:3], [0.299, 0.587, 0.114])
    if repDims:
        ret = np.zeros((G.shape[0], G.shape[1], 3))
        for k in range(3):
            ret[:, :, k] = G
        return ret
    else:
        return G

def cleanupTempFiles():
    files = os.listdir('.')
    for f in files:
        if f.find(TEMP_STR) > -1:
            os.remove(f)

#Input: path: Either a filename or a folder
#Returns: tuple (Video NxP array, dimensions of video)
def loadVideo(path, YCbCr = False):
    if not os.path.exists(path):
        print("ERROR: Video path not found: %s"%path)
        return None
    #Step 1: Figure out if path is a folder or a filename
    prefix = "%s/"%path
    isFile = False
    if os.path.isfile(path):
        isFile = True
        #If it's a filename, use avconv to split it into temporary frame
        #files and load them in
        prefix = TEMP_STR
        command = [AVCONV_BIN,
                    '-i', path,
                    '-f', 'image2',
                    TEMP_STR + '%d.png']
        subprocess.call(command)

    #Step 2: Load in frame by frame
    #First figure out how many images there are
    #Note: Frames are 1-indexed
    NFrames = 0
    while True:
        filename = "%s%i.png"%(prefix, NFrames+1)
        if os.path.exists(filename):
            NFrames += 1
        else:
            break
    if NFrames == 0:
        print("ERROR: No frames loaded")
        return (None, None)
    F0 = mpimage.imread("%s1.png"%prefix)
    IDims = F0.shape
    #Now load in the video
    I = np.zeros((NFrames, F0.size))
    print("Loading video.")
    for i in range(NFrames):
        if i%20 == 0:
            print(".")
        filename = "%s%i.png"%(prefix, i+1)
        IM = mpimage.imread(filename)
        if YCbCr:
            IM = rgb2ntsc(IM)
        I[i, :] = IM.flatten()
        if isFile:
            #Clean up temporary files
            os.remove(filename)
    print("\nFinished loading %s"%path)
    return (I, IDims)

#Returns: tuple (Video NxP array, dimensions of video)
def loadCVVideo(path, show_video=False):
    if not os.path.exists(path):
        print("ERROR: Video path not found: %s"%path)
        return None
    import cv2
    videoReader = cv2.VideoCapture(path)
    NFrames = int(videoReader.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    idx = 0
    AllFrames = np.array([])
    while videoReader.isOpened():
        validity,frame = videoReader.read()
        if frame == None:
            break
        IDims = frame.shape
        if AllFrames.size == 0:
            AllFrames = np.zeros((NFrames,frame.size))
        AllFrames[idx,:] = frame.flatten()
        idx += 1
        # optionally show it as we load it
        if show_video:
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
            #if cv2.waitKey(1) & 0xff == ord('q'):
                #break
    videoReader.release()
    if show_video:
        cv2.destroyAllWindows()
    return (AllFrames, IDims)

def loadImageIOVideo(path,pyr_level=0):
    if not os.path.exists(path):
        print("ERROR: Video path not found: %s"%path)
        return None
    import imageio
    if pyr_level > 0:
        from skimage.transform import pyramid_gaussian

    videoReader = imageio.get_reader(path, 'ffmpeg')
    NFrames = videoReader.get_length()
    I,I_feat = None,None
    for i in range(0, NFrames):
        frame = videoReader.get_data(i)
        feat_frame = np.array(frame)
        if pyr_level > 0:
            feat_frame = tuple(pyramid_gaussian(frame, pyr_level, downscale = 2))[-1]
        if I is None:
            I = np.zeros((NFrames, frame.size))
            I_feat = np.zeros((NFrames, feat_frame.size))
            IDims = frame.shape
        I[i, :] = np.array(frame.flatten(), dtype = np.float32)/255.0
        I_feat[i, :] = np.array(feat_frame.flatten(), dtype = np.float32)/255.0
    return (I, I_feat, IDims)

def loadVideoResNetFeats(path, depth=0):
    if not os.path.exists(path):
        print("ERROR: Video path not found: %s"%path)
        return None

    import imageio
    import torch
    import torch.nn as nn
    import torch.backends.cudnn as cudnn
    import torchvision.transforms as transforms
    import torchvision.models as models
    from torch.autograd import Variable

    # get first layers of network - we can try out different things here
    if depth == 0:
        shallow_layers = ['conv1','bn1','relu','maxpool']
    elif depth == 1:
        shallow_layers = ['conv1','bn1','relu','maxpool','layer1']
    else:
        print('unknown depth?',depth)
        shallow_layers = ['conv1','bn1','relu','maxpool']
    net = models.__dict__['resnet18'](pretrained=True)
    resnet_module = net.modules().__next__()
    resnet_modules = resnet_module.named_children()
    shallow_nn = []
    for module_name,module in resnet_modules:
        if module_name in shallow_layers:
            print('module ',module_name)
            shallow_nn.append(module)

    # preprocessing for network
    full_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    videoReader = imageio.get_reader(path, 'ffmpeg')
    NFrames = videoReader.get_length()
    I,I_feat = None,None
    for i in range(0, NFrames):
        frame = videoReader.get_data(i)
        frame_th = full_transform(Image.fromarray(frame)) # easiest to convert to PIL, then apply torch transforms
        frame_mb = frame_th.view(1,frame_th.size()[0],frame_th.size()[1],frame_th.size()[2])
        frame_var = Variable(frame_mb,requires_grad=False)
        for module in shallow_nn:
            frame_var = module(frame_var)
        th_data = frame_var.data
        single_th_data = th_data.view(th_data.size()[1],th_data.size()[2],th_data.size()[3])
        feat_frame = single_th_data.numpy()
        if I is None:
            I = np.zeros((NFrames, frame.size))
            I_feat = np.zeros((NFrames, feat_frame.size))
            IDims = frame.shape
        I[i, :] = np.array(frame.flatten(), dtype = np.float32)/255.0
        I_feat[i, :] = np.array(feat_frame.flatten(), dtype = np.float32)
    return (I, I_feat, IDims)

def loadVideoFolder(foldername):
    N = len(os.listdir(foldername))
    #Assume numbering starts at zero
    f0 = scipy.misc.imread("%s/%i.png"%(foldername, 0))
    IDims = f0.shape
    dim = len(f0.flatten())
    I = np.zeros((N, dim))
    I[0, :] = np.array(f0.flatten(), dtype=np.float32)/255.0
    for i in range(1, N):
        f = scipy.misc.imread("%s/%i.png"%(foldername, i))
        I[i, :] = np.array(f.flatten(), dtype=np.float32)/255.0
    return (I, IDims)

def saveFrames(I, IDims, frame_dir='frames/'):
    for idx in range(I.shape[0]):
        frame = np.reshape(I[idx,:],IDims)
        rescaled_frame = (255.0*frame).astype(np.uint8)
        Image.fromarray(rescaled_frame).save(frame_dir+'frame-'+str(idx)+'.jpg')

#Output video
#I: PxN video array, IDims: Dimensions of each frame
def saveVideo(I, IDims, filename, FrameRate = 30, YCbCr = False, Normalize = False):
    #Overwrite by default
    if os.path.exists(filename):
        os.remove(filename)
    N = I.shape[0]
    if YCbCr:
        for i in range(N):
            frame = np.reshape(I[i, :], IDims)
            I[i, :] = ntsc2rgb(frame).flatten()
    if Normalize:
        I = I-np.min(I)
        I = I/np.max(I)
    for i in range(N):
        frame = np.reshape(I[i, :], IDims)
        frame[frame < 0] = 0
        frame[frame > 1] = 1
        mpimage.imsave("%s%i.png"%(TEMP_STR, i+1), frame)
    if os.path.exists(filename):
        os.remove(filename)
    #Convert to video using avconv
    command = [AVCONV_BIN,
                '-r', "%i"%FrameRate,
                '-i', TEMP_STR + '%d.png',
                '-r', "%i"%FrameRate,
                '-b', '30000k',
                filename]
    subprocess.call(command)
    #Clean up
    for i in range(N):
        os.remove("%s%i.png"%(TEMP_STR, i+1))


#############################################################
####        SLIDING WINDOW VIDEO TOOLS, GENERAL         #####
#############################################################
def getPCAVideo(I):
    ICov = I.dot(I.T)
    [lam, V] = linalg.eigh(ICov)
    lam[lam < 0] = 0
    V = V*np.sqrt(lam[None, :])
    return V

def getSlidingWindowVideo(I, dim, Tau, dT):
    N = I.shape[0] #Number of frames
    P = I.shape[1] #Number of pixels (possibly after PCA)
    pix = np.arange(P)
    NWindows = int(np.floor((N-dim*Tau)/dT))
    X = np.zeros((NWindows, dim*P))
    idx = np.arange(N)
    for i in range(NWindows):
        idxx = dT*i + Tau*np.arange(dim)
        start = int(np.floor(idxx[0]))
        end = int(np.ceil(idxx[-1]))
        f = scipy.interpolate.interp2d(pix, idx[start:end+1], I[idx[start:end+1], :], kind='linear')
        X[i, :] = f(pix, idxx).flatten()
    return X

def getSlidingWindowVideoInteger(I, dim):
    N = I.shape[0]
    M = N-dim+1
    X = np.zeros((M, I.shape[1]*dim))
    for i in range(X.shape[0]):
        X[i, :] = I[i:i+dim, :].flatten()
    return X

def getTimeDerivative(I, Win):
    dw = np.floor(Win/2)
    t = np.arange(-dw, dw+1)
    sigma = 0.4*dw
    xgaussf = t*np.exp(-t**2  / (2*sigma**2))
    #Normalize by L1 norm to control for length of window
    xgaussf = xgaussf/np.sum(np.abs(xgaussf))
    xgaussf = xgaussf[:, None]
    IRet = scipy.signal.convolve2d(I, xgaussf, 'valid')
    validIdx = np.arange(dw, I.shape[0]-dw, dtype='int64')
    return [IRet, validIdx]


#############################################################
####            FAST TIME DELAY EMBEDDING, Tau = 1      #####
#############################################################
#Input: I: P x N Video with frames along the columns
#W: Windows
#Ouput: Mu: P x W video with mean frames along the columns
def tde_mean(I, W):
    IOut = np.array(I)
    IOut[IOut > 1] = 1
    IOut[IOut < 0] = 0
    start_time = time.time()
    N = I.shape[1]
    P = I.shape[0]
    Mu = np.zeros((P, W))
    for i in range(W):
        Mu[:, i] = np.mean(I[:, np.arange(N-W+1) + i], 1)
    end_time = time.time()
    print("tde_mean elapsed time ", end_time-start_time, " seconds, I.shape = ", I.shape, ", W = ", W)
    return Mu

#Frames assumed to be in each column
#Stacked frames are also in one column
#The delay frames are in a matrix I call "ID" which is never explicitly
#stored
#Return a tuple of (right hand singular vectors, singular values)
def tde_rightsvd(I, W, Mu):
    start_time = time.time()
    N = I.shape[1] #Number of frames in the video

    ## Step 1: Precompute frame and mean correlations
    B = I.T.dot(I);
    MuFlat = Mu.flatten()
    MuFlat = np.reshape(MuFlat, [len(MuFlat), 1])
    MuTMu = MuFlat.T.dot(MuFlat)
    C = Mu.T.dot(I) #A WxN matrix

    ## Step 2: Use precomputed information to compute (ID-Mu)^T*(ID-Mu)
    #Compute the ID^TID part
    ND = N-W+1
    IDTID = np.zeros((ND, ND))
    #Use the fact that a delay embedding is just a moving average along
    #all diagonals
    for i in range(N-W+1):
        b = np.diag(B, i)
        b2 = np.cumsum(b)
        bend = b2[W-1:]
        bbegin = np.zeros(len(bend))
        bbegin[1:] = b2[0:len(bend)-1]
        b2 = bend - bbegin
        IDTID[np.arange(len(b2)), i + np.arange(len(b2))] = b2
    IDTID = IDTID + IDTID.T
    np.fill_diagonal(IDTID, 0.5*np.diag(IDTID)) #Main diagonal was counted twice

    #Compute the Mu^TID part to subtract off mean
    MuTID = np.zeros((1, ND))
    for i in range(ND):
        MuTID[0, i] = np.sum(np.diag(C, i))
    ATA = IDTID - MuTID
    ATA = ATA - MuTID.T
    ATA = ATA + MuTMu
    #Handle numerical precision issues and keep it symmetric
    ATA = 0.5*(ATA + ATA.T)

    ## Step 3: Compute right singular vectors
    [S, Y] = linalg.eigh(ATA)
    idx = np.argsort(-S)
    S[S < 0] = 0 #Numerical precision
    S = np.sqrt(S[idx])
    Y = Y[:, idx]
    end_time = time.time()
    return (Y, S)

def getGradientVideo(I, IDims, sigma = 1):
    GV = np.zeros(I.shape)
    for i in range(I.shape[0]):
        X = np.reshape(I[i, :], IDims)
        G = rgb2gray(X, False)
        GM = gaussian_gradient_magnitude(G, sigma)
        F = np.zeros(IDims)
        for k in range(F.shape[2]):
            F[:, :, k] = GM
        GV[i, :] = F.flatten()
    return GV

def makeRandomWalkCurve(res, NPoints, dim):
    #Enumerate all neighbors in hypercube via base 3 counting between [-1, 0, 1]
    Neighbs = np.zeros((3**dim, dim))
    Neighbs[0, :] = -np.ones((1, dim))
    idx = 1
    for ii in range(1, 3**dim):
        N = np.copy(Neighbs[idx-1, :])
        N[0] += 1
        for kk in range(dim):
            if N[kk] > 1:
                N[kk] = -1
                N[kk+1] += 1
        Neighbs[idx, :] = N
        idx += 1
    #Exclude the neighbor that's in the same place
    Neighbs = Neighbs[np.sum(np.abs(Neighbs), 1) > 0, :]

    #Pick a random starting point
    X = np.zeros((NPoints, dim))
    X[0, :] = np.random.choice(res, dim)
    
    #Trace out a random path
    for ii in range(1, NPoints):
        prev = np.copy(X[ii-1, :])
        N = np.tile(prev, (Neighbs.shape[0], 1)) + Neighbs
        #Pick a random next point that is in bounds
        idx = np.sum(N > 0, 1) + np.sum(N < res, 1)
        N = N[idx == 2*dim, :]
        X[ii, :] = N[np.random.choice(N.shape[0], 1), :]
    return X

def smoothCurve(X, Fac):
    import scipy.interpolate as interp
    NPoints = X.shape[0]
    dim = X.shape[1]
    idx = range(NPoints)
    idxx = np.linspace(0, NPoints, NPoints*Fac)
    Y = np.zeros((NPoints*Fac, dim))
    NPointsOut = 0
    for ii in range(dim):
        Y[:, ii] = interp.spline(idx, X[:, ii], idxx)
        #Smooth with box filter
        y = (0.5/Fac)*np.convolve(Y[:, ii], np.ones(Fac*2), mode='same')
        Y[0:len(y), ii] = y
        NPointsOut = len(y)
    Y = Y[0:NPointsOut-1, :]
    Y = Y[2*Fac:-2*Fac, :]
    return Y

def getRandomMotionBlurMask(extent):
    from skimage.draw import line
    X = makeRandomWalkCurve(40, 20, 2)
    Y = smoothCurve(X, 20)
    Y = Y - np.mean(Y, 0)[None, :]
    Y = Y/np.max(Y, 0)
    Y = Y*extent
    theta = np.random.rand()*2*np.pi
    Y[:, 0] = Y[:, 0] + np.cos(theta)*np.linspace(0, extent, Y.shape[0])
    Y[:, 1] = Y[:, 1] + np.sin(theta)*np.linspace(0, extent, Y.shape[0])
    D = np.sum(Y**2, 1)[:, None]
    D = D + D.T - 2*Y.dot(Y.T)
    D[D < 0] = 0
    D = 0.5*(D + D.T)
    D = np.sqrt(D)
    Y = Y*extent/np.max(D)
    Y = Y - np.mean(Y, 0)[None, :]
    Y = Y - np.min(Y)
    I = np.zeros((extent, extent))
    for i in range(Y.shape[0]-1):
        c = [Y[i, 0], Y[i, 1], Y[i+1, 0], Y[i+1, 1]]
        c = [int(np.round(cc)) for cc in c]
        rr, cc = line(c[0], c[1], c[2], c[3])
        rr = [min(max(rrr, 0), extent-1) for rrr in rr]
        cc = [min(max(ccc, 0), extent-1) for ccc in cc]
        I[rr, cc] += 1.0
    I = I/np.sum(I)
    return (Y, I)

def simulateCameraShake(I, IDims, shakeMag):
    J = np.zeros(I.shape)
    for i in range(J.shape[0]):
        print("Blurring frame %i of %i"%(i, J.shape[0]))
        X = np.reshape(I[i, :], IDims)
        (_, mask) = getRandomMotionBlurMask(shakeMag)
        IBlur = 0*X
        for k in range(X.shape[2]):
            IBlur[:, :, k] = scipy.signal.fftconvolve(X[:, :, k], mask, 'same')
        #IBlur = np.array(IBlur, dtype=np.uint8)
        J[i, :] = IBlur.flatten()
    return J

if __name__ == '__main__2':
    (I, IDims) = loadVideo("VocalCordsVideos/LTR_ED_MucusBiphonCrop.avi")
    IBlur = simulateCameraShake(I, IDims, 40)
    saveVideo(IBlur, IDims, "MucusBlur.avi")
    IGradient = getGradientVideo(I, IDims, sigma=1)
    IGradient = IGradient/np.max(IGradient)
    saveVideo(IGradient, IDims, "out.avi")

if __name__ == '__main__2':
    (I, IDims) = loadVideo("Videos/heartcrop.avi")
    for lam in [0.01, 0.05, 0.1, 0.2, 0.5]:
        (IRet, IDimsRet) = simulateByteErrors(I, IDims, lam)
        saveVideo(IRet, IDimsRet, "VideoCorrupted%.2g.ogg"%lam)

if __name__ == '__main__':
    #(I, IDims) = make2ShakingPulses(400, T1 = 10, T2 = 10*np.pi/3, A1 = 20, A2 = 20, ydim = 160)
    (I, IDims) = make2GaussianPulses(400, T1 = 10, T2 = 10*np.pi/3, ydim = 160)
    saveVideo(I, IDims, "QuasiperiodicPulses.ogg")
