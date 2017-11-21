# SloMoLoops


## Getting Started

You can checkout the code
~~~~~ bash
git clone https://github.com/ctralie/SloMoLoops.git
cd SloMoLoops
git submodule update --init
git submodule update --remote
~~~~~

If this worked properly, you should see a subdirectory "ripser" and a subdirectory "SlidingWindowVideoTDA" both populated with files.  You will also need to compile the C files used for ripser

~~~~~ bash
cd ripser
python setup.py build_ext --inplace
~~~~~



[Chris Tralie]: <http://www.ctralie.com>
