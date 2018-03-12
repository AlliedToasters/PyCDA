import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import completeness_score

class ErrorAnalyzer(object):
    """Error Analyzer is used to measure predictive performance
    of cda. It is intended for use on images where all craters
    have been hand labeled and so are "known". The PyCDA prediction
    object accepts known craters as a pandas dataframe under the
    attribute .known_craters; the columns 'x', 'y', and 'diameter'
    should be populated with approprate values for known crater
    objects for use with ErrorAnalyzer.
    """
    def __init__(self):
        """An analyzer tracks predicted and known craters after
        performing analysis. Will reset after new call to .analyze
        """
        self.predicted = None
        self.known = None
        self.fp = 0
        self.tp = 0
        self.fn = 0
        self.D = None
        self.B = None
        self.Q = None
        self.done = False
        self.prediction_object = None
        pass

    def _match_predictions(self, prop, known):
        """Uses nearest neighbors algorithm to match
        known craters to proposals.
        """
        threshold = self.prediction_object.threshold
        nnb_search = NearestNeighbors(  
            n_neighbors=10, 
            algorithm='ball_tree', 
            metric='l2'
        )
        #fit to proposals to search this list
        nnb_search.fit(prop[['x', 'y', 'diameter']])
        distances, indices = nnb_search.kneighbors(known[['x', 'y', 'diameter']])

        #will keep results organized
        results = known.copy()
        cols1 = ['neighbor{}'.format(n) for n in range(10)]
        distances = pd.DataFrame(columns = cols1, data=distances)
        cols2 = ['id{}'.format(n) for n in range(10)]
        ids = pd.DataFrame(columns = cols2, data=indices)
        results = pd.concat([results, distances, ids], axis=1)

        #Scale distances by known crater diameter
        for col in cols1:
            results[col] = results[col]/results['diameter']

        #These copies will be our outputs
        known_copy = known[['x', 'y', 'diameter']].copy()
        prop_copy = prop[['x', 'y', 'diameter', 'likelihood']].copy()

        #initialize truth values
        known_copy['detected'] = False
        prop_copy['positive'] = False

        #iterate over neighbors, starting with nearest
        for n in range(10):
            results_col = 'neighbor{}'.format(n)
            prop_col = 'id{}'.format(n)
            #order by vicinity and iterate in ascending order
            for i, row in results.sort_values(results_col).iterrows():
                prop_id = int(row.loc[prop_col])
                #stop iteration once we hit threshold
                if row[results_col] > .4:
                    break
                #if crater/proposal haven't been matched, match them
                if not known_copy.at[i, 'detected'] and not prop_copy.at[prop_id, 'positive']:
                    #iff proposal was accepted by classifier
                    if prop_copy.at[prop_id, 'likelihood'] > threshold:
                        known_copy.at[i, 'detected'] = True
                        prop_copy.at[prop_id, 'positive'] = True
        
        print('found {} proposals to be true.'.format(len(prop_copy[prop_copy['positive']])))
        print('found {} craters to be detected.'.format(len(known_copy[known_copy['detected']])))
        return prop_copy, known_copy

    def print_report(self):
        """Prints performance statistics for prediction."""
        if not self.done:
            print('Call .analyze() on a prediction to get stats.')
            return None
        self.tp = len(self.predicted[self.predicted.positive])
        self.fp = len(self.predicted) - self.tp
        self.fn = len(self.known) - len(self.known[self.known.detected])
        self.done = True
        self.D = 100 * self.tp/(self.tp + self.fn)
        try:
            self.B = self.fp/self.tp
        except ZeroDivisionError:
            self.B = 99999
        self.Q = 100 * self.tp / (self.tp + self.fp + self.fn)
        print('='*50)
        print('\nDetection Percentage: %{}'.format(self.D), '\n')
        print('\nBranching Factor: ', self.B, '\n')
        print('\nQuality Percentage: %{}'.format(self.Q), '\n')
        print('='*50)
        return
    

    def analyze(self, prediction):
        """Takes a prediction object and performs analysis on it.
        Raises an exception if no known crater labels are attributed
        to the input prediction object.
        """
        if len(prediction.known_craters) == 0:
            raise Exception('Known crater statistics are required to perform '
                            'error analysis. Please populate the prediction object '
                            '.known_craters attribute with known crater locations '
                            "(pandas dataframe with columns 'x', y', 'diameter')")
        elif not isinstance(prediction.known_craters, type(pd.DataFrame())):
            #If data is passed in as an array, create
            #pandas dataframe for compatability. Assumes
            #data is ordered as in df; x(0), y(1), diameter(2)
            known_craters = prediction.known_craters
            df = pd.DataFrame(columns=['x', 'y', 'diameter'])
            df['x'] = known_craters[:, 0]
            df['y'] = known_craters[:, 1]
            df['diameter'] = known_craters[:, 2]
            craters = df
        elif isinstance(prediction.known_craters, type(pd.DataFrame())):
            craters = prediction.known_craters
        else:
            raise Exception('Sorry, labeled craters datatype is not understood. '
                            'Please populate the prediction object '
                            '.known_craters attribute with known crater locations '
                            "(pandas dataframe with columns 'x', y', 'diameter')")
        self.prediction_object = prediction
        self.predicted, self.known = self._match_predictions(
                                        prediction.proposals,
                                        craters
        )
        print('Matching complete!\n')
        self.done = True
        self.print_report()
        return

    def plot_densities(self):
        """Generates histogram plot with predicted and actual
        crater densities by size."""
        predictions = self.predicted[self.predicted.likelihood > self.prediction_object.threshold]
        min_ = min(min(self.known.diameter), min(predictions.diameter))
        max_ = max(max(self.known.diameter), max(predictions.diameter))
        bins = [n*(max_-min_)/20 + min_ for n in range(20)]
        fig, ax = plt.subplots();
        ax.hist(self.known.diameter.astype(np.float32), bins=bins, color='b', label='known crater density', alpha=.5);
        ax.hist(predictions.diameter.astype(np.float32), bins=bins, color='r', label='detected crater density', alpha=.5);
        ax.set_xlabel('crater diameter (pixels)');
        ax.set_ylabel('density over image area');
        plt.legend();
        plt.show();
        print('known crater count in image: ', len(self.known))
        print('detected crater count in image: ', len(predictions))
        return predictions
        
    def show(self):
        """Displays the input image with predictions and
        known craters displayed by colors.
        """
        image = self.prediction_object.input_image
        name = self.prediction_object.__name__
        threshold = self.prediction_object.threshold
        predictions = self.predicted[self.predicted.likelihood > threshold]
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image, cmap='Greys')
        ax.set_title('Detection performance for {}'.format(name))
        for i, crater in self.known.iterrows():
            if crater.detected:
                x = crater[0]
                y = crater[1]
                r = crater[2]/2
                circle = plt.Circle((x, y), r, fill=False, color='green');
                ax.add_artist(circle);
            elif not crater.detected:
                x = crater[0]
                y = crater[1]
                r = crater[2]/2
                circle = plt.Circle((x, y), r, fill=False, color='red');
                ax.add_artist(circle);
        for i, proposal in predictions.iterrows():
            if not proposal.positive:
                x = proposal[0]
                y = proposal[1]
                r = proposal[2]/2
                circle = plt.Circle((x, y), r, fill=False, color='yellow');
                ax.add_artist(circle);
        handles = []
        handles.append(mpatches.Patch(color='green', label='properly detected craters'))
        handles.append(mpatches.Patch(color='red', label='undetected craters'))
        handles.append(mpatches.Patch(color='yellow', label='false detections (noncraters detected as craters)'))
        plt.legend(handles=handles);
        plt.show();
        return

    def return_results(self):
        """Returns matched lists of known craters and detections."""
        return self.predicted, self.known

    
        

        
