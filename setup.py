from setuptools import setup, find_packages

setup(name='pycda',
    version='0.1.11',
    description='Python Crater Detection Algorithm (PyCDA) is a pipeline for crater detection; go from image to annotated crater stats in minutes.',
    url='https://github.com/AlliedToasters/PyCDA',
    download_url='https://github.com/AlliedToasters/PyCDA/archive/0.1.11.tar.gz',
    keywords = ['crater detection', 'astronomy', 'planetary science', 'planetary geology'],
    author='Michael Klear',
    author_email='michael.klear@colorado.edu',
    license='MIT',
    packages=['pycda'],
    classifiers=[],
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
