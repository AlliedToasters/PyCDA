import numpy as np
import pandas as pd
from skimage import measure
from math import sqrt, pi

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
    def __call__(self, image):
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
    
    def get_label_map(self, detection_map, threshold=.5):
        """Takes a pixel-wise prediction map and returns a matrix
        of unique objects on the map. Threshold is a hyperparameter
        for crater/non-crater pixel determination. Higher threshold
        may help distinguish merged crater detections.
        """
        filtered = np.where(detection_map > threshold, 1, 0)
        labels = measure.label(filtered, neighbors=4, background=0)
        return labels

    def get_crater_pixels(self, label_matrix, idx):
        """Takes a label matrix and a number and gets all the
        pixel locations from that crater object.
        """
        result = np.argwhere(np.where(label_matrix==idx, 1, 0))
        return result

    def get_pixel_objects(self, label_matrix):
        """Takes the label matrix and returns a list of objects.
        Each element in the list is a unique object, defined
        by an array of pixel locations belonging to it.
        """
        objects = []
        idx = 1
        result = np.array([0])
        while True:
            result = self.get_crater_pixels(label_matrix, idx)
            if len(result) == 0:
                break
            objects.append(result)
            idx += 1
        return objects

    def get_crater_proposals(self, detection_map):
        """Takes a pixel-wise prediction map and returns a list of
        crater proposals as x, y, d.
        """
        label_matrix = self.get_label_map(detection_map)
        proposals = self.get_pixel_objects(label_matrix)
        result = []
        for proposal in proposals:
            area = len(proposal)
            y_locs = [x[0] for x in proposal]
            x_locs = [x[1] for x in proposal]
            x_mean = round(np.mean(x_locs))
            y_mean = round(np.mean(y_locs))
            d = 2*sqrt(area/pi)
            result.append((x_mean, y_mean, d))
        return result
    
    
    def __call__(self, detection_map):
        cols = ['x', 'y', 'diameter']
        result = self.get_crater_proposals(detection_map)
        proposals = pd.DataFrame(columns = cols, data=result)
        proposals['likelihood'] = 1
        return proposals
    

def get(identifier):
    """handles argument to CDA pipeline for extractor specification.
    returns an initialized extractor.
    """
    model_dictionary = {
        'dummy': DummyExtractor,
        'circle': CircleExtractor
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
        
