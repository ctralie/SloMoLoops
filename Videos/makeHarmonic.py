import numpy as np
import sys
sys.path.append("..")
from VideoTools import *

def makeHarmonicVideo(filename):
    I, _, IDims = loadImageIOVideo(filename)
    INew = []
    for i in range(I.shape[0]):
        F1 = np.reshape(I[i, :], IDims)
        F2 = np.reshape(I[i*2%I.shape[0], :], IDims)
        F = np.concatenate((F1, F2), 1)
        INew.append(F.flatten())
    IDims = (IDims[0], IDims[1]*2, IDims[2])
    return (np.array(INew), IDims)

if __name__ == '__main__':
    (I, IDims) = makeHarmonicVideo('SlowMotionTemplateSimple.avi')
    saveVideo(I, IDims, "harmonic.avi")