gedfrom setuptools import setup, find_packages

setup(name='pycda',
    version='0.1.1',
    description='Python Crater Detection Algorithm (PyCDA) is a pipeline for crater detection; go from image to annotated crater stats in minutes.',
    url='https://github.com/AlliedToasters/PyCDA',
    download_url='https://github.com/AlliedToasters/PyCDA/archive/0.1.1.tar.gz',
    keywords = ['crater detection', 'astronomy', 'planetary science', 'planetary geology'],
    author='Michael Klear',
    author_email='michael.klear@colorado.edu',
    license='MIT',
    packages=['pycda'],
    classifiers=[],
    install_requires=[
        'numpy==1.14.1',
        'pandas==0.22.0',
        'scikit-image==0.13.1',
        'scikit-learn==0.19.1',
        'h5py==2.7.1',
        'tensorflow==1.6.0',
        'Keras==2.1.5'
    ]
)
