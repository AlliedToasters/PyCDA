import numpy as np
import pandas as pd
from skimage import measure
from scipy import ndimage as ndi
from scipy.ndimage import find_objects
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
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
    
    
class FastCircles(ExtractorBaseClass):
    """Circle Extractor assumes all objects in detection map are
    circles. It identifies groups of pixels, computes
    their mean location (centroid), and diameter based on
    the number of pixels in the group. Takes a prediction object
    and returns a list of proposals.
    """
    def __init__(self, sensitivity=.5):
        """sensitivity is a hyperparameter that adjusts the extractor's
        sensitivity to pixels with smaller values; a higher sensitivity
        tends to yeild larger crater candidates, with a risk of merging
        adjacent craters. A lower sensitivity can exclude weak detections.
        """
        self.threshold = 1 - sensitivity
    
    def _get_label_map(self, detection_map, threshold=.5, verbose=False):
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

    def _get_crater_pixels(self, label_matrix, idx):
        """Takes a label matrix and a number and gets all the
        pixel locations from that crater object.
        """
        result = np.argwhere(np.where(label_matrix==idx, 1, 0))
        return result

    def _get_pixel_objects(self, label_matrix, verbose=False):
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

    def _get_crater_proposals(self, detection_map, verbose=False):
        """Takes a pixel-wise prediction map and returns a list of
        crater proposals as lat, long, diameter.
        """
        label_matrix = self._get_label_map(detection_map, verbose=verbose)
        proposals = self._get_pixel_objects(label_matrix, verbose=verbose)
        result = []
        if verbose:
            print('Defining proposals as circles...')
        for proposal in proposals:
            area = len(proposal)
            y_locs = [x[0] for x in proposal]
            x_locs = [x[1] for x in proposal]
            y_mean = round(np.mean(y_locs))
            x_mean = round(np.mean(x_locs))
            d = 2*sqrt(area/pi)
            if d > 4:
                result.append((y_mean, x_mean, d))
        if verbose:
            print('done!')
        return result
    
    
    def __call__(self, detection_map, verbose=False):
        cols = ['lat', 'long', 'diameter']
        result = self._get_crater_proposals(detection_map, verbose=verbose)
        proposals = pd.DataFrame(columns = cols, data=result)
        proposals['likelihood'] = 1
        return proposals
    
class WatershedCircles(ExtractorBaseClass):
    """Takes a prediction object after detection and returns a 
    list of crater proposals (by calling object on prediction.)
    Performs a 'watershed' analysis:
    -transforms image to binary for some threshold (default .5)
    -transforms pixel values to min distance to background (0) pixel
    -uses local maxima as points for 'water sources'
    -uses negative distance values to build 'basins'
    -fills 'basins' with water from sources
    -uses the boundaries where 'water meets' as segment boundaries
    -objects are converted to circles whose center is the centroid of the
    bounding box of the object, diameter is the mean(width, height) of bounding
    box.
    """
    def __init__(self, sensitivity=.5):
        """sensitivity is a hyperparameter that adjusts the extractor's
        sensitivity to pixels with smaller values.
        """
        self.threshold = 1 - sensitivity
    
    def _get_labels(self, detection_map, verbose=False):
        """Handles transformations and extracts labels
        for each object identified.
        """
        #Convert to binary pixel values
        binary = np.where(detection_map > self.threshold, 1, 0)
        #Distance transform
        distance = ndi.distance_transform_edt(binary)
        #Identify local maxima
        local_maxi = peak_local_max(distance, indices=False,
                                    labels=binary)
        #Get object labels
        markers = ndi.label(local_maxi)[0]
        labels = watershed(-distance, markers, mask=binary)
        
        return labels

    def _get_crater_proposals(self, detection_map, verbose=False):
        """Converts labeled objects into circle figures.
        """
        labels = self._get_labels(detection_map, verbose=verbose)
        objs = find_objects(labels)
        proposals = []
        for obj in objs:
            centy = (obj[0].stop+obj[0].start)/2
            centx = (obj[1].stop+obj[1].start)/2
            proposal = [centy, centx]
            dy = obj[0].stop - obj[0].start
            dx = obj[1].stop - obj[1].start
            diameter = np.mean([dy, dx])
            proposal.append(diameter)
            if diameter > 4:
                proposals.append(proposal)
        return proposals
    
    def __call__(self, detection_map, verbose=False):
        cols = ['lat', 'long', 'diameter']
        result = self._get_crater_proposals(detection_map, verbose=verbose)
        proposals = pd.DataFrame(columns = cols, data=result)
        proposals['likelihood'] = 1
        return proposals

class _DummyExtractor(ExtractorBaseClass):
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
    
    
def get(identifier):
    """handles argument to CDA pipeline for extractor specification.
    returns an initialized extractor.
    """
    model_dictionary = {
        'dummy': _DummyExtractor,
        'fast_circle': FastCircles,
        'watershed': WatershedCircles
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
        
