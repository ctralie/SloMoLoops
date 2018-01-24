import subprocess
import os
import numpy as np
from VideoReordering import *

Videos=['Videos/ChrisNeckAmp.avi','jumpingjacks2menlowres.ogg',\
        'Videos/face_results.mp4', 'Videos/baby_result.mp4', 'Videos/throat_mag_crop.mp4',
        'Videos/exercise_1_crop.mp4', 'Videos/exercise_2_crop.mp4']
LapOpt = ["--is-unweighted-laplacian", "--is-weighted-laplacian"]
MedianOpt = ["--is-simple-reorder", "--is-median-reorder"]
pyrLevel = 2

for V in Videos:
    for medianReorder in [0, 1]:
        for weighted in [0, 1]:
            for Kappa in [0, 0.05, 0.1, 0.15]:
                strs = [MedianOpt[medianReorder], LapOpt[weighted]]
                fileprefix = get_out_fileprefix('reordered', V, not medianReorder, weighted, False, pyrLevel, 0, Kappa=Kappa)
                print(fileprefix)
                if os.path.exists("%s.avi"%fileprefix):
                    print("Skipping %s..."%fileprefix)
                    continue
                cmd = ["python", "VideoReordering.py", "--filename", V, "--show-plots", "--pyr_level", "%i"%pyrLevel, "--Kappa", "%g"%Kappa] + strs
                subprocess.call(cmd)
