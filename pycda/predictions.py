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
        #tile_split_coords is a list of (x, y) coordinates
        #that map to every split necessary for the detector
        self.tile_split_coords = []
        #In the case that the detector output is different
        #from the input, destination coordinates for the output
        #are stored as det_split_coords
        self.det_split_coords = []
        #list of bools recording which predictions have been made.
        self.detections_made = np.array([False])
        #prediction map will record the outputs of detector
        self.detection_map = np.zeros((self.input_image.shape[0], self.input_image.shape[1]))
        #proposals will be stored here.
        self.proposals = pd.DataFrame(columns=['x', 'y', 'diameter', 'likelihood'])
        #optional latitude/longitude attribute if user wants to
        #specify position of image
        self.lat_long = None
        #optional scale if user wants metric crater sizes
        self.scale = None
        
    def __str__(self):
        return self.__name__
        
    def record_detection(self, detection, index):
        """Records a detection in the prediction map.
        Uses index to determine location of detection.
        """
        xmin = self.det_split_coords[index][1]
        xmax = min(xmin+detection.shape[1], self.detection_map.shape[1])
        ymin = self.det_split_coords[index][0]
        ymax = min(ymin+detection.shape[0], self.detection_map.shape[0])
        self.detection_map[ymin:ymax, xmin:xmax] = detection
        
        
    def get_proposals(self, threshold = .5):
        """Returns a dataframe of detected craters.
        Threshold determines a cutoff for proposal likelihood.
        """
        print('{} proposals in list.'.format(len(self.proposals)))
        df = self.proposals[self.proposals.likelihood >= threshold]
        df = df[['x', 'y', 'diameter']].copy()
        print('Returning {} proposals.'.format(len(df)))
        return df
    
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
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.input_image)
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
                x = crater[0]
                y = crater[1]
                r = crater[2]/2
                circle = plt.Circle((x, y), r, fill=False, color='r');
                ax.add_artist(circle);
        plt.show();
        
    def show_detection(self):
        """Plots the detection map alongside the input image.
        """
        fig, ax = plt.subplots(ncols=2, figsize=(12, 8))
        if self.input_image.shape==2:
            cmap1 = 'Greys'
            ax[0].imshow(self.input_image, cmap=cmap1)
        else:
            ax[0].imshow(self.input_image)
        ax[1].imshow(self.detection_map)
        plt.show();
            