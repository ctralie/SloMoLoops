import numpy as np
from PIL import Image
from scipy import interpolate
import time
import argparse
import pyflow
import sys
import os

# Flow Options: taken from demo
alpha = 0.012
ratio = 0.75
minWidth = 20
nOuterFPIterations = 7
nInnerFPIterations = 1
nSORIterations = 30
colType = 0

all_flow_filenames = sys.argv[1:]
n_flow_fields = len(all_flow_filenames)//2
forward_flow_filenames = all_flow_filenames[:n_flow_fields]
backward_flow_filenames = all_flow_filenames[n_flow_fields:]
x0,x1 = None,None
tracks = None
t = 0
for forward_flow_filename,backward_flow_filename in zip(forward_flow_filenames,backward_flow_filenames):
    forward_flow = np.load(forward_flow_filename)
    backward_flow = np.load(backward_flow_filename)

    # populate sampling grid
    if x0 is None:
        x0 = np.arange(forward_flow.shape[0])
        x1 = np.arange(forward_flow.shape[1])
        tracks = np.zeros((n_flow_fields,x0.shape,x1.shape,2))
        for xdx in range(x0.shape):
            tracks[t,xdx,:,0] = blah
    forward_u = interpolate.interp2d(x0,x1,forward_flow[:,:,0])
    forward_v = interpolate.interp2d(x0,x1,forward_flow[:,:,1])
    backward_u = interpolate.interp2d(x0,x1,forward_flow[:,:,0])
    backward_v = interpolate.interp2d(x0,x1,forward_flow[:,:,1])
    print('flow shapes:',forward_flow.shape,backward_flow.shape)
