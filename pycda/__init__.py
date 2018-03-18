import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image
import gc
from pycda import detectors
from pycda import extractors
from pycda import classifiers
from pycda import predictions
from pycda import util_functions
from pycda.util_functions import update_progress, resolve_color_channels, get_steps, get_crop_specs, crop_array, remove_ticks, make_batch

class CDA(object):
    """the CDA class is a pipeline that makes predictions
    on an input image by passing it through a series of
    models. For each prediction, it generates a prediction
    object that tracks the outputs of the various models throughout
    the process.
    """
    def __init__(self, detector='tiny', extractor='fast_circle', classifier='convolution'):
        """To initialize the CDA, you must define a detector, extractor, and
        classifier. You can pass these in as arguments or use text aliases
        to specify.
        """
        #initialize the models.
        self.detector = detectors.get(detector)
        if isinstance(extractor, list):
            self.extractor = [extractors.get(ex) for ex in extractor]
        else:
            self.extractor = [extractors.get(extractor)]
        self.classifier = classifiers.get(classifier)
        #track previous predictions.
        self.predictions = []

    def _get_prediction(self, input_image):
        """Checks to see if prediction has been made on the same image.
        If not found, creates a new prediction object.
        """
        for prediction in self.predictions:
            if np.array_equal(prediction.input_image, input_image):
                return prediction
            else:
                pass
        new_prediction = predictions.Prediction(input_image, len(self.predictions), self)
        self.predictions.append(new_prediction)
        return new_prediction

    def _prepare_detector(self, prediction):
        """Gets prediction object ready for use by
        the detector by populating coordinate lists.
        """
        if prediction.verbose:
            print('Preparing detection steps...')
        
        #Calculate latitude steps
        height = prediction.input_image.shape[0]
        yin = self.detector.input_dims[0]
        yout = self.detector.output_dims[0]
        y_steps_in, y_steps_out = get_steps(height, yin, yout)
        
        #Calculate longitude steps
        width = prediction.input_image.shape[1]
        xin = self.detector.input_dims[1]
        xout = self.detector.output_dims[1]
        x_steps_in, x_steps_out = get_steps(width, xin, xout)
        
        #iterate through every step and record in prediction object
        for ystep in zip(y_steps_in, y_steps_out):
            for xstep in zip(x_steps_in, x_steps_out):
                #Record ordered positional steps for input (lat, long)
                prediction.image_split_coords.append((ystep[0], xstep[0]))
                #and for output (lat, long)
                prediction.det_split_coords.append((ystep[1], xstep[1]))
                
        #set all predictions status to False
        prediction.detections_made = np.full((len(prediction.image_split_coords)), False, dtype=bool)
        if prediction.verbose:
            print('Done!\nDetection will require {} steps'.format(len(prediction.detections_made)))
        return prediction
    
    def _batch_detect(self, prediction, batch_size=None, verbose=False):
        """Generates batches to feed to detector,
        gets detection maps, handles bookkeeping.
        Returned prediction object has a completed detection
        map.
        """
        if verbose:
            print('Performing detections...')
        #determine batch size
        if batch_size == None:
            batch_size = self.detector.rec_batch_size
        image = util_functions.resolve_color_channels(prediction, self.detector)
        crop_dims = self.detector.input_dims
        #rescale color values for model if necessary
        if image.dtype == np.uint8:
            image = image/255
        while any(~prediction.detections_made):
            if verbose:
                progress = prediction.detections_made.sum()
                progress = progress/len(prediction.detections_made)
                update_progress(progress)
            #Find next index range for detection
            first_index = prediction.detections_made.sum()
            remaining_predictions = len(prediction.detections_made) - first_index
            last_index = min(first_index+batch_size, first_index+remaining_predictions)
            #Record index range in slice object
            indices = slice(first_index, last_index)
            #Get cropping coordinates
            crop_coords = prediction.image_split_coords[indices]
            #Build batch and predict
            batch = make_batch(image, crop_dims, crop_coords)
            results = self.detector.predict(batch)
            #Record detections to prediction object
            indices_enumerated = range(indices.start, indices.stop)
            prediction._batch_record_detection(results, indices_enumerated)
            prediction.detections_made[indices] = True
        #delete duplicate image for memory management.
        if verbose:
            update_progress(1)
        del image
        return prediction
    
    def _batch_classify(self, prediction, batch_size=None, verbose=False):
        """Performs batch classifications on crater proposals.
        Updates the likelihood values for prediction proposals."""
        if verbose:
            print('performing classifications...')
        #determine batch size
        if batch_size == None:
            batch_size = self.classifier.rec_batch_size
        dim = self.classifier.input_dims
        df = prediction.proposals
        iter_row = df.iterrows()
        image = resolve_color_channels(prediction, self.classifier)
        #tracks all results
        likelihoods = []
        #Will switch when iteration is through
        done = False
        while not done:
            #records cropping coords for batch maker
            crops = []
            crop_dims = []
            while len(crops) < batch_size:
                try:
                    i, row = next(iter_row)
                    if verbose:
                        progress = i/len(df)
                        update_progress(progress)
                except StopIteration:
                    done = True
                    break
                proposal = row[['lat', 'long', 'diameter']].values
                crop_orgn, crop_dim = get_crop_specs(proposal, self.classifier)
                crops.append(crop_orgn)
                crop_dims.append(crop_dim)
            batch = make_batch(image, crop_dims, crops, out_dims=dim)
            results = self.classifier.predict(batch)
            likelihoods += [result[0] for result in results]
        prediction.proposals['likelihood'] = likelihoods
        #delete temporary image from memory
        del image
        return prediction

    def _predict(self, input_image, verbose=False):
        """Calls a series of functions to perform prediction on input
        image. Returns a prediction object.
        """
        if isinstance(input_image, CDAImage):
            input_image = input_image.as_array()
        elif isinstance(input_image, type(np.array([0]))):
            pass
        else:
            input_image = np.array(input_image)
        prediction = self._get_prediction(input_image)
        if np.all(prediction.detections_made):
            if verbose:
                print('detections already made!')
            prediction.proposals = pd.DataFrame(columns=['lat', 'long', 'diameter', 'likelihood'])
        else:
            prediction = self._prepare_detector(prediction)
        prediction = self._batch_detect(prediction, verbose=verbose)
        for ext in self.extractor:
            result = ext(prediction.detection_map, verbose=verbose)
            prediction.proposals = pd.concat([prediction.proposals, result], axis=0)
        #Reset proposal indices
        prediction.proposals.index = range(len(prediction.proposals))
        if verbose:
            print(
                len(prediction.proposals), 
                ' proposals extracted from detection map.\n'
            )
        prediction = self._batch_classify(prediction, verbose=verbose)
        if verbose:
            print(
                '\n',
                np.where(prediction.proposals.likelihood > prediction.threshold, 1, 0).sum(),
                ' objects classified as craters.\n'
            )
        return prediction
    
    def predict(self, input_image, threshold=.5, verbose=False):
        """Intended for 'out of box' use. Calls predictions
        and returns a pandas dataframe with crater predictions.
        """
        prediction = self._predict(input_image, verbose=verbose)
        return prediction._predict(threshold=threshold)
    
    def get_prediction(self, input_image, verbose=False):
        """Used for accessing the prediction object.
        Calls predictions and returns the prediction
        object for advanced statistics and visualizations.
        """
        return self._predict(input_image, verbose=verbose)
    
class CDAImage(object):
    """Special image object for CDA; Stored as an array,
    but with .show() function for viewing.
    """
    def __init__(self, image):
        #Common use case: convert from array to PIL image
        if isinstance(image, type(np.array([0]))):
            self.image = image
        #In case another CDAImage object passed in
        elif isinstance(image, type(self)):
            self.image = image.image
        #In case PIL image passed in
        elif isinstance(image, type(Image.new('1', (1,1)))):
            self.image = np.array(image)
        else:
            raise Exception('Image object constructor does not'
                            ' understand input image type.')
     
    def show(self, show_ticks=False):
        """Displays the input image using PIL image object"""
        fig, ax = plt.subplots();
        ax.imshow(self.image, cmap='Greys_r');
        if not show_ticks:
            ax = remove_ticks(ax)
        plt.show();
        return None
    
    def as_array(self):
        """If array version of image is needed."""
        return self.image
    
def load_image(filename):
    """load an image from input filepath and return
    a numpy array image."""
    image = io.imread(filename)
    try:
        assert isinstance(image, type(np.array([0])))
    except AssertionError:
        raise Exception('Could not load file into numpy array.')
    return CDAImage(image)

