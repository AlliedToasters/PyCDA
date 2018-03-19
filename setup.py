from setuptools import setup, find_packages
import pkg_resources

path = pkg_resources.resource_filename('pycda', 'models/tinynet.h5')

setup(name='pycda',
    version='0.1.13',
    description='Python Crater Detection Algorithm (PyCDA) is a pipeline for crater detection; go from image to annotated crater stats in minutes.',
    url='https://github.com/AlliedToasters/PyCDA',
    download_url='https://github.com/AlliedToasters/PyCDA/archive/0.1.13.tar.gz',
    keywords = ['crater detection', 'astronomy', 'planetary science', 'planetary geology'],
    author='Michael Klear',
    author_email='michael.klear@colorado.edu',
    license='MIT',
    packages=find_packages(),
    classifiers=[],
    data_files = [
        'pycda/models/tinynet.h5',
        'pycda/models/classifier_12x12_2.h5',
        'pycda/sample_imgs/holdout_tile_labels.csv',
        'pycda/sample_imgs/holdout_tile.pgm',
        'pycda/sample_imgs/rgb_sample.jpg',
    ],
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'scikit-image',
        'scikit-learn',
        'h5py',
        'tensorflow',
        'Keras'
    ]
)
