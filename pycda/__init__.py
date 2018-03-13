import numpy as np
from skimage import io
from PIL import Image
import gc
from pycda import detectors
from pycda import extractors
from pycda import classifiers
from pycda import predictions
from pycda import util_functions
from pycda.util_functions import update_progress

class CDA(object):
    """the CDA class is a pipeline that makes predictions
    on an input image by passing it through a series of
    models. For each prediction, it generates a prediction
    object that tracks the outputs of the various models throughout
    the process.
    """
    def __init__(self, detector='tiny', extractor='circle', classifier='convolution'):
        """To initialize the CDA, you must define a detector, extractor, and
        classifier. You can pass these in as arguments or use text aliases
        to specify.
        """
        #initialize the models.
        self.detector = detectors.get(detector)
        self.extractor = extractors.get(extractor)
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
        width = prediction.input_image.shape[1]
        xin = self.detector.input_dims[1]
        xout = self.detector.output_dims[1]
        x_steps_in, x_steps_out = util_functions.get_steps(width, xin, xout)
        
        #Repeat for y dimension
        height = prediction.input_image.shape[0]
        yin = self.detector.input_dims[0]
        yout = self.detector.output_dims[0]
        y_steps_in, y_steps_out = util_functions.get_steps(height, yin, yout)
        
        #iterate through every step and record in prediction object
        for ystep in zip(y_steps_in, y_steps_out):
            for xstep in zip(x_steps_in, x_steps_out):
                prediction.tile_split_coords.append((ystep[0], xstep[0]))
                prediction.det_split_coords.append((ystep[1], xstep[1]))
                
        #set all predictions status to False
        prediction.detections_made = np.full((len(prediction.tile_split_coords)), False, dtype=bool)
        if prediction.verbose:
            print('Done!\nDetection will require {} steps'.format(len(prediction.detections_made)))
        return prediction
    
    def _batch_detect(self, prediction, batch_size=None):
        """Generates batches to feed to detector,
        gets detection maps, handles bookkeeping.
        Returned prediction object has a completed detection
        map.
        """
        #determine batch size
        if batch_size == None:
            batch_size = self.detector.rec_batch_size
        #exit loop after all detections made
        n_channels = self.detector.input_channels
        #copy image to preserve color value
        image = prediction.input_image.copy()
        image = util_functions.resolve_color_channels(image, desired=n_channels)
        #rescale color values for model if necessary
        if image.dtype == np.uint8:
            image = image/255
        if prediction.verbose:
            print('performing detections...')
        while any(~prediction.detections_made):
            batch = []
            first_index = prediction.detections_made.sum()
            remaining_predictions = len(prediction.detections_made) - first_index
            last_index = min(first_index+batch_size, first_index+remaining_predictions)
            indices = range(first_index, last_index)
            for index in indices:
                if prediction.verbose:
                    progress = index
                    progress = progress/len(prediction.detections_made)
                    update_progress(progress)
                crop_coords = prediction.tile_split_coords[index]
                crop_dims = self.detector.input_dims
                next_image = util_functions.crop_array(image, crop_dims[0], crop_dims[1], crop_coords)
                if len(next_image.shape) == 2:
                    #add color channel to greyscale image
                    next_image = np.expand_dims(next_image, axis=-1)
                batch.append(next_image)
            batch = np.array(batch)
            results = self.detector.predict(batch)
            batch_index = 0
            for index in indices:
                result = results[batch_index, :, :, 0]
                prediction.record_detection(result, index)
                prediction.detections_made[index] = True
                batch_index += 1
        #delete temporary image from memory
        del image
        if prediction.verbose:
            update_progress(1.)
        return prediction
    
    def _batch_classify(self, prediction, batch_size=None):
        """Performs batch classifications on crater proposals.
        Updates the likelihood values for prediction proposals."""
        #determine batch size
        if batch_size == None:
            batch_size = self.classifier.rec_batch_size
        df = prediction.proposals
        iter_row = df.iterrows()
        n_channels = self.classifier.input_channels
        #copy image to preserve color value
        image = prediction.input_image.copy()
        image = util_functions.resolve_color_channels(image, desired=n_channels)
        #tracks all results
        likelihoods = []
        #track for cropping call
        crop_dims = self.classifier.input_dims
        crater_size = self.classifier.crater_pixels
        #Will switch when iteration is through
        if prediction.verbose:
            print('Performing classifications...')
        done = False
        while not done:
            #will become our batch
            batch = []
            while len(batch) < batch_size:
                try:
                    i, row = next(iter_row)
                except StopIteration:
                    done = True
                    break
                if prediction.verbose:
                    progress = i/len(df)
                    update_progress(progress)
                proposal = row[['x', 'y', 'diameter']].values
                cropped = util_functions.crop_crater(image, proposal, dim=crop_dims, px=crater_size)
                if len(cropped.shape) == 2:
                    cropped = np.expand_dims(cropped, axis=-1)
                batch.append(cropped)
            batch = np.array(batch)
            results = self.classifier.predict(batch)
            likelihoods += [result[0] for result in results]
        prediction.proposals['likelihood'] = likelihoods
        #delete temporary image from memory
        del image
        if prediction.verbose:
            update_progress(1.)
            print('\n')
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
        prediction.verbose = verbose
        if np.all(prediction.detections_made):
            if verbose:
                print('prediction already made! returning...')
            return prediction
        prediction = self._prepare_detector(prediction)
        prediction = self._batch_detect(prediction)
        prediction.proposals = self.extractor(prediction.detection_map, verbose=verbose)
        if verbose:
            print(
                len(prediction.proposals), 
                ' proposals extracted from detection map.\n'
            )
        prediction = self._batch_classify(prediction)
        if verbose:
            print(
                np.where(prediction.proposals.likelihood > prediction.threshold, 1, 0).sum(),
                ' objects classified as craters.\n'
            )
        return prediction
    
    def predict(self, input_image, threshold=.5, verbose=False):
        """Intended for 'out of box' use. Calls predictions
        and returns a pandas dataframe with crater predictions.
        """
        prediction = self._predict(input_image)
        return prediction.get_proposals(threshold=threshold, verbose=verbose)
    
    def get_prediction(self, input_image, verbose=False):
        """Used for accessing the prediction object.
        Calls predictions and returns the prediction
        object for advanced statistics and visualizations.
        """
        return self._predict(input_image, verbose=verbose)
    
class CDAImage(object):
    """Special image object for CDA. Just a PIL image object
    that returns an array version when needed
    """
    def __init__(self, image):
        #Common use case: convert from array to PIL image
        if isinstance(image, type(np.array([0]))):
            self.image = Image.fromarray(image)
        #In case another CDAImage object passed in
        elif isinstance(image, type(self)):
            self.image = image.image
        #In case PIL image passed in
        elif isinstance(image, type(Image.new('1', (1,1)))):
            self.image = image
        else:
            raise Exception('Image object constructor does not'
                            ' understand input image type.')
     
    def show(self, size='reasonable'):
        """Displays the input image using PIL image object"""
        if size == 'reasonable':
            input_x, input_y = self.image.size
            if input_x > 600:
                output_x = 600
            else:
                output_x = input_x
            if input_y > 600:
                output_y = 600
            else:
                output_x = input_x
            image = self.image.resize((output_x, output_y))
        else:
            image = self.image
        image.show()
        return
    
    def as_array(self):
        """If array version of image is needed."""
        return np.array(self.image)
    
def load_image(filename):
    """load an image from input filepath and return
    a numpy array image."""
    image = io.imread(filename)
    try:
        assert isinstance(image, type(np.array([0])))
    except AssertionError:
        raise Exception('Could not load file into numpy array.')
    return CDAImage(image)

