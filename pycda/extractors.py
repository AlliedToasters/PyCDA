import numpy as np
import pandas as pd
from skimage import measure
from scipy.ndimage import find_objects
from math import sqrt, pi
from pycda.util_functions import update_progress

class ExtractorBaseClass(object):
    """Base class for an extractor object. The extractor
    converts a prediction map into a list of crater candidates.
    """
    def __init__(self):
        pass
    
    def __call__(self):
        raise Exception('Extractor base class cannot perform extraction.')
        
class DummyExtractor(ExtractorBaseClass):
    """Dummy Extractor takes an input image and returns a list of
    random predictions for testing. Proposals are a list of tuples.
    Each tuple in the list has the crater proposal as (in pixels):
    (x position, y position, diameter).
    """
    def __call__(self, image, verbose=False):
        width = image.shape[0]
        height = image.shape[1]
        n_proposals = np.random.randint(2,50)
        proposals = []
        for prop in range(n_proposals):
            x_pos = np.random.randint(0, width)
            y_pos = np.random.randint(0, height)
            diameter = np.random.randint(2, width+height/20)
            likelihood = 1
            proposals.append((x_pos, y_pos, diameter, 1))
        proposals = pd.DataFrame(columns=['x', 'y', 'diameter', 'likelihood'], data=proposals)
        if verbose:
            print('I am a dummy extractor!')
        return proposals

class CircleExtractor(ExtractorBaseClass):
    """Circle Extractor assumes all objects in detection map are
    circles. It identifies groups of pixels, computes
    their mean location (centroid), and diameter based on
    the number of pixels in the group.
    """
    def __init__(self, sensitivity=.5):
        """sensitivity is a hyperparameter that adjusts the extractor's
        sensitivity to pixels with smaller values; a higher sensitivity
        tends to yeild larger crater candidates, with a risk of merging
        adjacent craters. A lower sensitivity can exclude weak detections.
        """
        self.threshold = 1 - sensitivity
    
    def get_label_map(self, detection_map, threshold=.5, verbose=False):
        """Takes a pixel-wise prediction map and returns a matrix
        of unique objects on the map. Threshold is a hyperparameter
        for crater/non-crater pixel determination. Higher threshold
        may help distinguish merged crater detections.
        """
        if verbose:
            print('getting label map...')
        filtered = np.where(detection_map > threshold, 1, 0)
        labels = measure.label(filtered, neighbors=4, background=0)
        if verbose:
            print('done!')
        return labels

    def get_crater_pixels(self, label_matrix, idx):
        """Takes a label matrix and a number and gets all the
        pixel locations from that crater object.
        """
        result = np.argwhere(np.where(label_matrix==idx, 1, 0))
        return result

    def get_pixel_objects(self, label_matrix, verbose=False):
        """Takes the label matrix and returns a list of objects.
        Each element in the list is a unique object, defined
        by an array of pixel locations belonging to it.
        """
        objects = []
        idx = 1
        result = np.array([0])
        if verbose:
            print('Getting proposals...')
            end = np.max(label_matrix)
            progress = 0
        while True:
            result = self.get_crater_pixels(label_matrix, idx)
            if verbose:
                update_progress(progress/end)
                progress += 1
            if len(result) == 0:
                break
            objects.append(result)
            idx += 1
        return objects

    def get_crater_proposals(self, detection_map, verbose=False):
        """Takes a pixel-wise prediction map and returns a list of
        crater proposals as x, y, d.
        """
        label_matrix = self.get_label_map(detection_map, verbose=verbose)
        proposals = self.get_pixel_objects(label_matrix, verbose=verbose)
        result = []
        if verbose:
            print('Defining proposals as circles...')
        for proposal in proposals:
            area = len(proposal)
            y_locs = [x[0] for x in proposal]
            x_locs = [x[1] for x in proposal]
            x_mean = round(np.mean(x_locs))
            y_mean = round(np.mean(y_locs))
            d = 2*sqrt(area/pi)
            result.append((x_mean, y_mean, d))
        if verbose:
            print('done!')
        return result
    
    
    def __call__(self, detection_map, verbose=False):
        cols = ['x', 'y', 'diameter']
        result = self.get_crater_proposals(detection_map, verbose=verbose)
        proposals = pd.DataFrame(columns = cols, data=result)
        proposals['likelihood'] = 1
        return proposals
    
    
class FastCircles(ExtractorBaseClass):
    """Performs the same task as CircleExtractor,
    but ostensibly faster.
    """
    def __init__(self, sensitivity=.5):
        """sensitivity is a hyperparameter that adjusts the extractor's
        sensitivity to pixels with smaller values; a higher sensitivity
        tends to yeild larger crater candidates, with a risk of merging
        adjacent craters. A lower sensitivity can exclude weak detections.
        """
        self.threshold = 1 - sensitivity
    
    def get_label_map(self, detection_map, threshold=.5, verbose=False):
        """Takes a pixel-wise prediction map and returns a matrix
        of unique objects on the map. Threshold is a hyperparameter
        for crater/non-crater pixel determination. Higher threshold
        may help distinguish merged crater detections.
        """
        if verbose:
            print('getting label map...')
        filtered = np.where(detection_map > threshold, 1, 0)
        labels = measure.label(filtered, neighbors=4, background=0)
        if verbose:
            print('done!')
        return labels

    def get_crater_pixels(self, label_matrix, idx):
        """Takes a label matrix and a number and gets all the
        pixel locations from that crater object.
        """
        result = np.argwhere(np.where(label_matrix==idx, 1, 0))
        return result

    def get_pixel_objects(self, label_matrix, verbose=False):
        """Takes the label matrix and returns a list of objects.
        Each element in the list is a unique object, defined
        by an array of pixel locations belonging to it.
        """
        objects = find_objects(label_matrix)
        result = []
        for prop in objects:
            slice_ = np.argwhere(label_matrix[prop])
            slice_[:, 0] += prop[0].start
            slice_[:, 1] += prop[1].start
            result.append(slice_)
        return result

    def get_crater_proposals(self, detection_map, verbose=False):
        """Takes a pixel-wise prediction map and returns a list of
        crater proposals as x, y, d.
        """
        label_matrix = self.get_label_map(detection_map, verbose=verbose)
        proposals = self.get_pixel_objects(label_matrix, verbose=verbose)
        result = []
        if verbose:
            print('Defining proposals as circles...')
        for proposal in proposals:
            area = len(proposal)
            y_locs = [x[0] for x in proposal]
            x_locs = [x[1] for x in proposal]
            x_mean = round(np.mean(x_locs))
            y_mean = round(np.mean(y_locs))
            d = 2*sqrt(area/pi)
            result.append((x_mean, y_mean, d))
        if verbose:
            print('done!')
        return result
    
    
    def __call__(self, detection_map, verbose=False):
        cols = ['x', 'y', 'diameter']
        result = self.get_crater_proposals(detection_map, verbose=verbose)
        proposals = pd.DataFrame(columns = cols, data=result)
        proposals['likelihood'] = 1
        return proposals

def get(identifier):
    """handles argument to CDA pipeline for extractor specification.
    returns an initialized extractor.
    """
    model_dictionary = {
        'dummy': DummyExtractor,
        'circle': CircleExtractor,
        'fast_circle': FastCircles
    }
    if identifier is None:
        raise Exception('You must specify a proposal extractor.')
    if isinstance(identifier, ExtractorBaseClass):
        model = identifier
        return model
    elif identifier in model_dictionary:
        return model_dictionary[identifier]()
    elif callable(identifier):
        if isinstance(identifier(), DetectorBaseClass):
            return identifier()
        else:
            raise Exception('custom extractors must inherit'
                           'from ExtractorBaseClass, which can be'
                           'imported from extractors.py')
    else:
        raise ValueError('Could not interpret '
                         'extractor identifier: {} \n'
                         'try one of these keywords: {}'
                         .format(identifier, list(model_dictionary.keys())))
        
