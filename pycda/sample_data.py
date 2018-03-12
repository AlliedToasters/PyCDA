from pycda import load_image
import pandas as pd
import pkg_resources

def get_sample_image(filename='holdout_tile.pgm'):
    """Retrieves sample data from the in-package directory."""
    path = pkg_resources.resource_filename('pycda', 'sample_imgs/{}'.format(filename))
    img = load_image(path)
    return img

def get_sample_csv(filename='holdout_tile_labels.csv'):
    """Retrieves hand-labeled crater annotations"""
    path = pkg_resources.resource_filename('pycda', 'sample_imgs/{}'.format(filename))
    df = pd.read_csv(path)
    return df
