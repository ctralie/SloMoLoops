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
from scipy.ndimage import gaussian_gradient_magnitude

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

def loadImageIOVideo(path):
    if not os.path.exists(path):
        print("ERROR: Video path not found: %s"%path)
        return None
    import imageio
    videoReader = imageio.get_reader(path, 'ffmpeg')
    NFrames = videoReader.get_length()
    F0 = videoReader.get_data(0)
    IDims = F0.shape
    I = np.zeros((NFrames, F0.size))
    I[0, :] = np.array(F0.flatten(), dtype = np.float32)/255.0
    for i in range(1, NFrames):
        I[i, :] = np.array(videoReader.get_data(i).flatten(), dtype = np.float32)/255.0
    return (I, IDims)

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
