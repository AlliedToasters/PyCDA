import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycda import util_functions

class Prediction(object):
    """A prediction object is a specialized data
    handler for pycda. It tracks the progress of predictions
    on an input image, helps the pipeline track information,
    and can perform auxiliary functions that help the user
    inspect the prediction, save the results, export csv files,
    and modify hyperparameters.
    """
    def __init__(self, image, id_no, cda):
        """prediction objects are initialized by the cda pipeline itself."""
        #the prediction object stores the input image in memory.
        self.input_image = image
        self.__name__ = 'prediction_{}'.format(id_no)
        self.cda = cda
        self.verbose=False
        #image_split_coords is a list of (x, y) coordinates
        #that map to every split necessary for the detector
        self.image_split_coords = []
        #In the case that the detector output is different
        #from the input, destination coordinates for the output
        #are stored as det_split_coords
        self.det_split_coords = []
        #list of bools recording which predictions have been made.
        self.detections_made = np.array([False])
        #prediction map will record the outputs of detector
        self.detection_map = np.zeros((self.input_image.shape[0], self.input_image.shape[1]))
        #proposals will be stored here.
        self.proposals = pd.DataFrame(columns=['lat', 'long', 'diameter', 'likelihood'])
        #threshold is a likelihood value below which proposals
        #are rejected. Lowering this value will include more proposals
        #in prediction, while raising it while be more selective.
        self.threshold = .5
        #add ground truth labels for errors module
        self.known_craters = pd.DataFrame(columns=['lat', 'long', 'diameter'])
        #optional scale if user wants metric crater sizes
        self.scale = None
        
    def __str__(self):
        return self.__name__
        
    def _record_detection(self, detection, index):
        """Records a detection in the prediction map.
        Uses index to determine location of detection.
        """
        ymin = self.det_split_coords[index][0]
        ymax = min(ymin+detection.shape[0], self.detection_map.shape[0])
        xmin = self.det_split_coords[index][1]
        xmax = min(xmin+detection.shape[1], self.detection_map.shape[1])
        self.detection_map[ymin:ymax, xmin:xmax] = detection
        
    def _batch_record_detection(self, batch, indices):
        """Takes a batch of detections and a slice object
        that contains first, last index of batch. Records
        detections into detection map.
        """
        for i, index in enumerate(indices):
            detection = batch[i, :, :, 0]
            self._record_detection(detection, index)
        return
        
    def _predict(self, threshold = .5):
        """Returns a dataframe of detected craters.
        Threshold determines a cutoff for proposal likelihood.
        """
        df = self.proposals[self.proposals.likelihood >= threshold]
        df = df[['lat', 'long', 'diameter']].copy()
        return df
    
    def get_proposals(self):
        return self.proposals
    
    def set_scale(self, scale):
        """User can set scale for statistics in meters.
        scale should be meters per pixel.
        """
        self.scale = scale
        
    
    def show(self, threshold=.5, include_ticks=True):
        """Displays the input image with the predicted craters
        overlaid. Threshold determines the likelihood for which a proposal
        should be displayed.
        """
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(self.input_image, cmap='Greys_r')
        ax.set_title('Crater detections for {}'.format(self.__name__))
        if include_ticks:
            if self.scale == None:
                message = '(in pixels, resolution unspecified)'
            else:
                message = '@ {} meters/pixel'.format(self.scale)
            ax.set_ylabel('horizontal distance {}'.format(message))
            ax.set_xlabel('vertical direction {}'.format(message))
        else:
            ax = util_functions.remove_ticks(ax)
        for i, crater in self.proposals.iterrows():
            if crater.likelihood > threshold:
                x = crater[1]
                y = crater[0]
                r = crater[2]/2
                circle = plt.Circle((x, y), r, fill=False, color='r');
                ax.add_artist(circle);
        plt.show();
        
    def show_detection(self, remove_ticks=True):
        """Plots the detection map alongside the input image."""
        fig, ax = plt.subplots(ncols=2, figsize=(9, 6))
        ax[0].imshow(self.input_image, cmap='Greys_r')
        ax[1].imshow(self.detection_map, cmap='CMRmap')
        if remove_ticks:
            ax[0], ax[1] = util_functions.remove_ticks(ax[0]), util_functions.remove_ticks(ax[1])
        plt.show();
        
    def to_csv(self, filepath, likelihoods=False, index=False):
        """Creates a csv file with predictions. If likelihoods
        is True, a likelihoods column is added to the csv file.
        Saves csv to filepath usind pd.to_csv method."""
        if len(self.proposals) == 0:
            print('Cannot export csv. No predictions made!')
            return
        df = self.proposals
        if not likelihoods:
            df = df[['x', 'y', 'diameter']]
        df.to_csv(filepath, index=index)
        return
        
            
