# PyCDA: Simple Crater Detection
Go from image to crater annotations in minutes.

<b>PyCDA is a crater detection algorithm (CDA) written in Python.</b><br>

Inspired by research in applying convolutional neural networks to <a href='https://www.hou.usra.edu/meetings/lpsc2018/pdf/2202.pdf'>crater detection</a> (Benedix et al.) and <a href='https://arxiv.org/pdf/1601.00978.pdf'>crater candidate classification</a> (Cohen et al.), PyCDA is aimed at making CDA research modular and usable.<br><br>
The current release, pre-alpha "fun" 0.1.14, is a conceptual demonstration; its general performance on some datasets is too poor for use; however, it will yield crater detections.

## Getting Started

At its most basic level, PyCDA is built to be <I>easy to use</I>, and that should start with installation; pre-alpha "fun" version 0.1.14 is now available via PyPI with:

```
pip install pycda
```

### Prerequisites

PyCDA currently supports Python 3.6; we recommend using a virtual environment or environment manager such as <a href='https://conda.io/docs/user-guide/install/index.html#regular-installation'>conda</a>, as PyCDA has not been tested on previous versions of its dependencies.

### Installing

PyCDA's current release, "fun" 0.1.14, is a prototype pre-release. However, it is available for download via PyPi for the adventurous.
From your python 3.6 environment, install with pip via the command line:

```
pip install pycda
```

### Using PyCDA

For a quick prediction "out of the box," use the commands:

```
from pycda import CDA, load_image

cda = CDA()
image = load_image('my_image_filepath.png')
detections = cda.predict(image)
```

The output of the call to .predict is a pandas dataframe, with columns 'lat' (crater location from top of image), 'long' (crater location from left edge of image), and diameter' (crater diameter in pixels).

PyCDA currently handles image using PIL; image files from disc must therefore be in the formats that PIL supports. Numpy arrays of raster images are also supported; pass them in as you would an image object.

PyCDA provides visualization and error analysis tools as well; check out the <a href='https://github.com/AlliedToasters/PyCDA/blob/master/demo.ipynb'>demo notebook</a> for a peek at these features!

Documentation on the entire project is available <a href='http://pycda.readthedocs.io/en/latest/index.html'>here</a>.

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

PyCDA follows something like [SemVer](http://semver.org/) guidelines, the current release is "fun" 0.1.14 and is still in early development. I fixed the data file loading issues that came with 'super top secret pre-alpha release 0.1.1', and we finally have something that does something "out of the box."

## Authors

* **Michael Klear** - *Initial work* - [AlliedToasters](https://github.com/AlliedToasters)

## Contributing

PyCDA is a community project and we welcome anybody in the CDA research community, planetary scientists, or Python developers to the fold. Please reach out to Michael Klear at:<br>

michael.klear@colorado.edu<br>

-or-<br>

michael.r.klear@gmail.com<br>

to contribute!


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
