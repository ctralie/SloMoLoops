from VideoTools import *
from VideoReordering import *
from multiprocessing import Pool as PPool
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import argparse
import os

def getVideoPyramid(I, IDims, pyr_level):
    """
    Get a coarse level of the image pyramid for ever frame
    in lieu of the original video
    """
    if pyr_level == 0:
        return I
    from skimage.transform import pyramid_gaussian
    IRet = []
    for i in range(0, I.shape[0]):
        frame = np.reshape(I[i, :], IDims)
        feat_frame = tuple(pyramid_gaussian(frame, pyr_level, downscale = 2))[-1]
        IRet.append(feat_frame.flatten())
    return np.array(IRet)

def getVideoResNet(IOrig, IDims, depth):
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
    net.eval()
    resnet_module = net.modules().__next__()
    resnet_modules = resnet_module.named_children()
    shallow_nn = []
    for module_name,module in resnet_modules:
        if module_name in shallow_layers:
            shallow_nn.append(module)

    # preprocessing for network
    full_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    NFrames = IOrig.shape[0]
    IRet = []
    for i in range(0, NFrames):
        frame = np.array(255.0*np.reshape(IOrig[i, :], IDims), dtype = np.uint8)
        frame_th = full_transform(Image.fromarray(frame)) # easiest to convert to PIL, then apply torch transforms
        frame_mb = frame_th.view(1,frame_th.size()[0],frame_th.size()[1],frame_th.size()[2])
        frame_var = Variable(frame_mb,requires_grad=False)
        for module in shallow_nn:
            frame_var = module(frame_var)
        th_data = frame_var.data
        single_th_data = th_data.view(th_data.size()[1],th_data.size()[2],th_data.size()[3])
        feat_frame = single_th_data.numpy()
        IRet.append(np.array(feat_frame.flatten(), dtype = np.float32))
    return np.array(IRet)

def getCircleDist(x, y):
    return np.minimum(np.abs(x - y), 2*np.pi-np.abs(x - y))

def getKendallTauCircle(rank1, rank2):
    """
    Given two global rankings, return the Kendall Tau Score
    on the circle
    """
    N = len(rank1)
    r1 = rank1/float(N)
    r2 = rank2/float(N)
    diff = r1[None, :] - r1[:, None]
    A = np.sign(diff)
    A[np.abs(diff) > 0.5] *= -1 #Wrap around
    diff = r2[None, :] - r2[:, None]
    B = np.sign(diff)
    B[np.abs(diff) > 0.5] *= -1
    return np.sum(A*B)/float(N*(N-1))

def getCycles(NCycles, filename, NFinal = -1):
    I, I_feat, IDims = loadImageIOVideo(filename)
    N = I.shape[0]
    theta = np.linspace(0, 2*np.pi, N)
    idx = np.mod(NCycles*np.arange(N), N)
    if NFinal > 0:
        starti = np.random.randint(len(idx)-NFinal)
        idx = idx[starti:starti+NFinal]
    #Circularly permute indices
    idx = np.roll(idx, np.random.randint(len(idx)))
    theta = theta[idx]
    theta = np.unwrap(theta)
    theta = theta - np.min(theta)
    theta = np.mod(theta, 2*np.pi)
    return {'I':I[idx, :], 'IDims':IDims, 'idx':idx, 'theta':theta}

def doTest(filename, NCycles, noise = 0, shake = 0, bgfac = 0, fileprefix = "", Verbose = False, saveVideos = False):
    doPlot = False
    if len(fileprefix) > 1:
        doPlot = True
    doSimple = True
    res = getCycles(NCycles = NCycles, filename = filename)
    [I, IDims, thetagt] = [res['I'], res['IDims'], res['theta']]
    I += noise*np.random.randn(I.shape[0], I.shape[1])
    if shake > 0:
        simulateCameraShake(I, IDims, shake)
    if bgfac > 0:
        simulateBGBlob(I, IDims, bgfac)
    if saveVideos:
        saveVideo(I, IDims, "%s_simulated.avi"%fileprefix)
    ret = {}
    for pyr_level in [3]:#range(-2, 4):
        if Verbose:
            print("pyr_level = %i"%pyr_level)
        if pyr_level >= 0:
            I_feat = getVideoPyramid(I, IDims, pyr_level)
        else:
            depth = -pyr_level - 1
            I_feat = getVideoResNet(I, IDims, depth)
        for Alpha in [0, 0.25, 0.5, 0.75, 1.0]:
            for Weighted in [True]:
                for doSlidingWindow in [True, False]:
                    thisprefix = "%i_%i_%g_%i"%(doSlidingWindow, pyr_level, Alpha, Weighted)
                    res = reorderVideo(I, I_feat, IDims, derivWin = 0, Weighted = Weighted, \
                                        doSimple = doSimple, doPlot = doPlot, Verbose = Verbose, \
                                        doImageAnalogies = False, Alpha = Alpha, \
                                        fileprefix = fileprefix+thisprefix, returnAnswer = False, \
                                        doSlidingWindow = doSlidingWindow)
                    theta = np.mod(res['thetau'], 2*np.pi)
                    thetagt = thetagt[0:len(theta)]
                    theta = theta[0:len(thetagt)]
                    circErr = np.mean(getCircleDist(theta, thetagt))
                    rank1 = 0*thetagt
                    rank1[np.argsort(thetagt)] = np.arange(len(rank1))
                    rank2 = 0*rank1
                    rank2[np.argsort(theta)] = np.arange(len(rank2))
                    kendTau = getKendallTauCircle(rank1, rank2)
                    ret[thisprefix] = [circErr, kendTau]
                    if doPlot:
                        plt.figure(figsize=(8, 6))
                        plt.subplot(211)
                        plt.scatter(thetagt, theta)
                        plt.title("err = %.3g"%circErr)
                        plt.subplot(212)
                        plt.scatter(rank1, rank2)
                        plt.title("$\\tau = %.3g$"%kendTau)
                        plt.savefig("%s_CircCoordsCorr.svg"%(fileprefix+thisprefix), bbox_inches = 'tight')
                    if saveVideos and res['X']:
                        saveVideo(res['X'], IDims, (fileprefix+thisprefix)+".avi")
    return ret

def writeBatchHeader(fout, filename):
    
    fout.write("""<html>
    <body><h1>{}</h1>
    <table border = "1">
    <tr><td>NCycles</td><td>Noise</td><td>Shake</td><td>BgFac</td><td>TrialNumber</td><td>SlidingWindow</td>
        <td>PyramidLevel</td><td>Alpha</td><td>Weighted</td><td>CircError</td><td>KendallTau</td></tr>   
    """.format(filename))

def writeBatchHeaderCSV(fout, filename):
    fout.write("NCycles,Noise,Shake,BgFac,TrialNumber,SlidingWindow,PyramidLevel,Alpha,Weighted,CircError,KendallTau")

def doBatchTests(filename, fout, batchidx = -1):
    idx = 0
    noise = 0
    for NCycles in [3, 5, 10, 15, 20, 25, 30, 40, 50]:
        for (noise, shake, bgfac) in [(0, 0, 0), (0, 10, 0), (0, 20, 0), \
                                        (0, 30, 0), (0, 40, 0), (0, 50, 0),\
                                        (0.5, 0, 0), (1, 0, 0), (2, 0, 0),\
                                        (0, 0, 0.01), (0, 0, 0.02), (0, 0, 0.05)]:
            for trial in range(50):
                if idx == batchidx:
                    np.random.seed(idx)
                    ret = doTest(filename, NCycles, noise, shake, bgfac, Verbose = True)
                    for item in ret:
                        fout.write("%i,%g,%i,%g,%i,"%(NCycles, noise, shake, bgfac, trial))
                        fout.write(("%s,"*len(item.split("_")))%tuple(item.split("_")))
                        fout.write(("%g,%g\n")%tuple(ret[item]))
                        fout.flush()
                idx += 1

def doBatch(args):
    (filename, outdir, batchidx) = args
    print("Doing Batch %i..."%batchidx)
    csvfilename = "%s/%i.csv"%(outdir, batchidx)
    if not os.path.exists(csvfilename):
        fout = open(csvfilename, "w")
        doBatchTests(filename, fout, batchidx)
        fout.close()
    else:
        print("Skipping %s"%csvfilename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=0, help="batch")
    parser.add_argument('--nthreads', type=int, default=3, help="Number of threads in parallel pool")
    parser.add_argument('--out_dir', type=str, default=".", help="out directory")
    parser.add_argument('--videofile', type=str, default='Videos/SlowMotionTemplateCrowd.avi', help="Template video")
    parser.add_argument('--NCycles', type=int, default=0, help = "Number of cycles")
    parser.add_argument('--shake', type=int, default=0, help = "Shake by pixels")
    parser.add_argument('--noise', type=float, default=0, help = "AWGN coefficient")
    parser.add_argument('--bgfac', type=float, default=0, help = "Background blob area ratio")
    parser.add_argument('--makeplots', type=int, default=0, help='Save plots of circular coordinates to disk')
    parser.add_argument('--savevideos', type=int, default=0, help="Whether to save the simulated video")
    opt = parser.parse_args()

    filename = opt.videofile
    outdir = opt.out_dir
    batchidx = opt.batch
    
    if opt.NCycles > 0:
        NCycles = opt.NCycles
        noise = opt.noise
        shake = opt.shake
        bgfac = opt.bgfac
        saveVideos = opt.savevideos
        fileprefix = "%i_%g_%i"%(NCycles, noise, shake)
        trial = 0
        htmlfilename = "%s/%s.html"%(outdir, fileprefix)
        fout = open(htmlfilename, "w")
        writeBatchHeader(fout, filename)
        if not opt.makeplots:
            fileprefix = ""
        ret = doTest(filename, NCycles, noise, shake, bgfac, Verbose = True, fileprefix=fileprefix, saveVideos = saveVideos)
        for item in ret:
            fout.write("<tr>")
            fout.write("<td>%i</td><td>%g</td><td>%i</td><td>%g</td><td>%i</td>"%(NCycles, noise, shake, bgfac, trial))
            fout.write(("<td>%s</td>"*len(item.split("_")))%tuple(item.split("_")))
            fout.write(("<td>%g</td>"*2)%tuple(ret[item]))
            fout.write("</tr>\n")
            fout.flush()
        fout.close()
    else:
        if batchidx >= 0:
            doBatch((filename, outdir, batchidx))
        else:
            fout = open("header.csv", "w")
            writeBatchHeaderCSV(fout, filename)
            fout.close()
            NBatches = -batchidx
            args = zip([filename]*NBatches, [outdir]*NBatches, range(NBatches))
            parpool = PPool(opt.nthreads)
            parpool.map(doBatch, args)
