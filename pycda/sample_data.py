from pycda import load_image
import pandas as pd
import pkg_resources
import random
import os

def get_sample_image(filename='holdout_tile.pgm', choose=False):
    """Retrieves sample data from the in-package directory.
    if choose=True, randomly selects sample photo from package.
    """
    path = pkg_resources.resource_filename('pycda', 'sample_imgs/')
    if choose:
        choices = os.listdir(path)
        filename = random.choice(choices)
        while filename[-4:] == '.csv':
            filename = random.choice(choices)
    file = path + filename
    img = load_image(file)
    return img

def get_sample_csv(filename='holdout_tile_labels.csv'):
    """Retrieves hand-labeled crater annotations for image
    holdout_tile.pgm (default returned by get_sample_image()).
    Returns pandas dataframe.
    """
    path = pkg_resources.resource_filename('pycda', 'sample_imgs/{}'.format(filename))
    df = pd.read_csv(path)
    return df
