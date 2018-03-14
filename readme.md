# PyCDA: Simple Crater Detection
Go from image to crater annotations in minutes.

<b>PyCDA is a crater detection algorithm (CDA) written in Python.</b><br>

Inspired by research in applying convolutional neural networks to <a href='https://www.hou.usra.edu/meetings/lpsc2018/pdf/2202.pdf'>crater detection</a> (Benedix et al.) and <a href='https://arxiv.org/pdf/1601.00978.pdf'>crater candidate classification</a> (Cohen et al.), PyCDA is aimed at making CDA research modular and usable.<br><br>

## Getting Started

At its most basic level, PyCDA is built to be <I>easy to use</I>, and that should start with installation; as we are still in development, the only way to install is by cloning this repository and installing from the cloned directory.

### Prerequisites

PyCDA currently supports Python 3.6; we recommend using a virtual environment or environment manager such as <a href='https://conda.io/docs/user-guide/install/index.html#regular-installation'>conda</a>, as PyCDA has not been tested on previous versions of its dependencies.

### Installing

Clone this repository. In your python 3.6 environment, go to to the root directory of PyCDA. Run

```
pip install -r requirements.txt
```

If you'd prefer to install the dependencies manually, use:

```
pip install numpy==1.14.1
pip install pandas==0.22.0
pip install scikit-image==0.13.1
pip install scikit-learn==0.19.1
pip install h5py==2.7.1
pip install tensorflow==1.6.0
pip install Keras==2.1.5

```

and, finally, from the root directory of the cloned repository:

```
pip install .
```

If you'd like to install in development mode, use the -e flag so changes become active immediately.

## Running the tests

Test your installation with test.py:

```
python test.py
```

The test takes about a minute on my machine; this is because it's testing batch predictions on tensorflow models,
which takes some time. I'm working to reduce test time.<br>

The test runs a detection on a sample image and runs an error analysis on its predicitions. It tests the most basic pieces of PyCDA. A more robust testing suite is under development.


## Versioning

As per [SemVer](http://semver.org/) guidelines, the current (first development) version is 0.1.0 and is the only available version.

## Authors

* **Michael Klear** - *Initial work* - [AlliedToasters](https://github.com/alliedtoasters)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
