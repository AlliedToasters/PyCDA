# PyCDA: Simple Crater Detection
Go from image to crater annotations in minutes.

<b>PyCDA is a crater detection algorithm (CDA) written in Python.</b><br>

Inspired by research in applying convolutional neural networks to <a href='https://www.hou.usra.edu/meetings/lpsc2018/pdf/2202.pdf'>crater detection</a> (Benedix et al.) and <a href='https://arxiv.org/pdf/1601.00978.pdf'>crater candidate classification</a> (Cohen et al.), PyCDA is aimed at making CDA research modular and usable.<br><br>

## Getting Started

At its most basic level, PyCDA is built to be <I>easy to use</I>, and that should start with installation; as we are still in development, the only way to install is by cloning this repository and installing from the cloned directory.

### Prerequisites

PyCDA currently supports Python 3.6; we recommend using a virtual environment or environment manager such as <a href='https://conda.io/docs/user-guide/install/index.html#regular-installation'>conda</a>, as PyCDA has not been tested on previous versions of its dependencies.

### Installing

PyCDA's current release, 0.1.1, is very early and unstable. However, it is available for download via PyPi for the adventurous.
From your python 3.6 environment, install with pip via the command line:

```
pip install pycda
```

If you'd like to install in development mode, use the -e flag so changes become active immediately.

### Using PyCDA

For a quick prediction "out of the box," use the commands:

```
from pycda import CDA, load_image

cda = CDA()
image = load_image('my_image_filepath.img')
detections = cda.predict(image)
```

PyCDA provides visualization and error analysis tools as well; check out the <a href='https://github.com/AlliedToasters/PyCDA/blob/master/demo.ipynb'>demo notebook</a> for a peek at these features!

## Running the tests

Test your installation with test.py, available from this repo. With wget:

```
wget https://raw.githubusercontent.com/AlliedToasters/PyCDA/master/test.py
```

Then, run

```
python test.py
```


## Versioning

PyCDA follows something like [SemVer](http://semver.org/) guidelines, the current release is 0.1.1 and is still in early development. I like to call it 'super top secret pre-alpha release 0.1.1', but it's not really a secret. You'll get version 0.1.1 and its dependencies via pip install pycda.

## Authors

* **Michael Klear** - *Initial work* - [AlliedToasters](https://github.com/alliedtoasters)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
