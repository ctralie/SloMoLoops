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

We will now go through a brief example toggling on and off some of the most common features.  First, we'll start with this video, which is the file <a href = "http://www.ctralie.com/Research/SloMoLoops/jumpingjacksbg.ogg">jumpingjacksbg.ogg</a>:

<video controls>
  <source src='http://www.ctralie.com/Research/SloMoLoops/jumpingjacksbg.ogg' type="video/ogg">
Your browser does not support the video tag.
</video>

[Christopher Tralie]: <http://www.ctralie.com>
[Matthew Berger]: <https://matthewberger.github.io/>
[ripser python package]: <https://github.com/ctralie/ripser>
[pyTorch]: <http://pytorch.org/>
[imageio]: <http://imageio.readthedocs.io/en/latest/installation.html>
