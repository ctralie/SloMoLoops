# SloMoLoops


## Dependencies
* Numpy/Scipy/Matplotlib
* [ripser python package] for persistent homology
* [imageio] for efficient video I/O
* [pyTorch] (Optional) For using resnet features instead of an image pyramid for mitigating motion drift


## Running
To see all of the options, run the script as follows

~~~~~ bash
python VideoReordering.py --help
~~~~~

We will now go through a brief example toggling on and off some of the most common features.  First, we'll start with the video <a href = "http://www.ctralie.com/Research/SloMoLoops/JumpingJacks_Occlusions.avi">JumpingJacks_Occlusions.avi</a>, which is a video of two men doing jumping jacks with a drifting occluding object added

<video controls>
  <source src='http://www.ctralie.com/Research/SloMoLoops/jumpingjacksbg.ogg' type="video/ogg">
Your browser does not support the video tag.
</video>

Now, let's do a simple reordering where we shuffle the frames by their circular coordinates, using a weighted Laplacian

~~~~~ bash
python VideoReordering.py --filename JumpingJacks_Occlusions.avi --is-simple-reorder --is-weighted-laplacian --show-plots
~~~~~

We get the following result

<img src = "JumpingJacks_Occlusions-reordered-0-simple-weighted-img-0.gif">


The code also outputs the following plot, which gives more information about the TDA and the Laplacian circular coordinates

![Jumping jacks simple reordered](http://www.ctralie.com/Research/SloMoLoops/JumpingJacks_Occlusions-reordered-0-simple-weighted-img-0_CircCoords.svg)


The above result is choppy, so let's do a median voting instead now

~~~~~ bash
python VideoReordering.py --filename jumpingjacksbg.ogg --is-median-reorder --is-weighted-laplacian
~~~~~

[Christopher Tralie]: <http://www.ctralie.com>
[Matthew Berger]: <https://matthewberger.github.io/>
[ripser python package]: <https://github.com/ctralie/ripser>
[pyTorch]: <http://pytorch.org/>
[imageio]: <http://imageio.readthedocs.io/en/latest/installation.html>
