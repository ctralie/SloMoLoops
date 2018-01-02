import numpy as np
from PIL import Image
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

all_frame_filenames = sys.argv[1:]
for idx in np.arange(1,len(all_frame_filenames)):
    prior_frame_filename,cur_frame_filename = all_frame_filenames[idx-1],all_frame_filenames[idx]
    prior_frame,cur_frame = np.array(Image.open(prior_frame_filename)),np.array(Image.open(cur_frame_filename))
    prior_frame = prior_frame.astype(float) / 255.
    cur_frame = cur_frame.astype(float) / 255.

    # backward flow
    s = time.time()
    u, v, warped_img_back = pyflow.coarse2fine_flow(prior_frame, cur_frame, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,nSORIterations, colType)
    e = time.time()
    print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (e - s, prior_frame.shape[0], prior_frame.shape[1], prior_frame.shape[2]))
    backward_flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    flow_filename = 'flow/'+'backward_flow-'+str(idx)+'.npy'
    np.save(flow_filename, backward_flow)

    # forward flow
    s = time.time()
    u, v, warped_img_forward = pyflow.coarse2fine_flow(cur_frame, prior_frame, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,nSORIterations, colType)
    e = time.time()
    print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (e - s, prior_frame.shape[0], prior_frame.shape[1], prior_frame.shape[2]))
    forward_flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    flow_filename = 'flow/'+'forward_flow-'+str(idx)+'.npy'
    np.save(flow_filename, forward_flow)

    # warped images
    base_dir = os.path.dirname(prior_frame_filename)+'/'
    rescaled_warped_frame = (255.0*warped_img_back).astype(np.uint8)
    Image.fromarray(rescaled_warped_frame).save(base_dir+'warped_backward-'+str(idx)+'.jpg')
    rescaled_warped_frame = (255.0*warped_img_forward).astype(np.uint8)
    Image.fromarray(rescaled_warped_frame).save(base_dir+'warped_forward-'+str(idx)+'.jpg')

