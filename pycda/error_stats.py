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

    def _match_predictions(self, prop, known, verbose=True):
        """Uses nearest neighbors algorithm to match
        known craters to proposals.
        """
        threshold = self.prediction_object.threshold
        #Will search through ten nearest proposals
        #unless there are fewer than ten.
        kn = min(10, len(prop))
        nnb_search = NearestNeighbors(  
            n_neighbors=kn, 
            algorithm='ball_tree', 
            metric='l2'
        )
        #fit to proposals to search this list
        nnb_search.fit(prop[['lat', 'long', 'diameter']])
        distances, indices = nnb_search.kneighbors(known[['lat', 'long', 'diameter']])

        #will keep results organized
        results = known.copy()
        cols1 = ['neighbor{}'.format(n) for n in range(kn)]
        distances = pd.DataFrame(columns = cols1, data=distances)
        cols2 = ['id{}'.format(n) for n in range(kn)]
        ids = pd.DataFrame(columns = cols2, data=indices)
        results = pd.concat([results, distances, ids], axis=1)

        #Scale distances by known crater diameter
        for col in cols1:
            results[col] = results[col]/results['diameter']

        #These copies will be our outputs
        known_copy = known[['lat', 'long', 'diameter']].copy()
        prop_copy = prop[['lat', 'long', 'diameter', 'likelihood']].copy()

        #initialize truth values
        known_copy['detected'] = False
        prop_copy['positive'] = False

        #iterate over neighbors, starting with nearest
        for n in range(kn):
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
        
        if verbose:
            print('{} craters were properly detected.'.format(len(known_copy[known_copy['detected']])))
        return prop_copy, known_copy

    def compute_results(self):
        """Computes descriptive statistics about model performance.
        """
        if not self.done:
            print('No results to compute!')
            return None
        self.tp = len(self.predicted[self.predicted.positive])
        self.fp = np.where(self.predicted[~self.predicted.positive].likelihood > self.prediction_object.threshold, 1, 0).sum()
        self.fn = len(self.known[~self.known.detected])
        self.D = 100 * self.tp/(self.tp + self.fn)
        self.R = self.tp/(self.tp + self.fn)
        self.P = self.tp/(self.tp + self.fp)
        self.F1 = 2/((1/self.P) + (1/self.R))
        self.FD = 1 - self.P
        self.FNR = self.fn/(self.tp + self.fn)
        try:
            self.B = self.fp/self.tp
        except ZeroDivisionError:
            self.B = self.fp/.00001
        self.Q = 100 * self.tp / (self.tp + self.fp + self.fn)
        return None
        
    def print_report(self):
        """Prints performance statistics for prediction."""
        if not self.done:
            print('Call .analyze() on a prediction to get stats.')
            return None
        print('='*50)
        print('\nDetection Percentage: %{}'.format(round(self.D, 1)))
        print('\nPrecision: {}'.format(round(self.P, 2)))
        print('\nRecall: {}'.format(round(self.R, 2)))
        print('\nF1-Score: {}'.format(round(self.F1, 2)))
        print('\nFalse Discovery Rate: {}'.format(round(self.FD, 2)))
        print('\nFalse Negative Rate: {}'.format(round(self.FNR, 2)))
        print('\nBranching Factor: ', round(self.B, 2))
        print('\nQuality Percentage: %{}'.format(round(self.Q, 1)), '\n')
        print('='*50)
        return
    

    def analyze(self, prediction, verbose=True):
        """Takes a prediction object and performs analysis on it.
        Raises an exception if no known crater labels are attributed
        to the input prediction object.
        """
        if len(prediction.known_craters) == 0:
            raise Exception('Known crater statistics are required to perform '
                            'error analysis. Please populate the prediction object '
                            '.known_craters attribute with known crater locations '
                            "(pandas dataframe with columns 'lat', long', 'diameter')")
        elif not isinstance(prediction.known_craters, type(pd.DataFrame())):
            #If data is passed in as an array, create
            #pandas dataframe for compatability. Assumes
            #data is ordered as in df; x(0), y(1), diameter(2)
            known_craters = prediction.known_craters
            df = pd.DataFrame(columns=['lat', 'long', 'diameter'])
            df['lat'] = known_craters[:, 0]
            df['long'] = known_craters[:, 1]
            df['diameter'] = known_craters[:, 2]
            craters = df
        elif isinstance(prediction.known_craters, type(pd.DataFrame())):
            craters = prediction.known_craters
            cols = craters.columns
            #attempts to format unlabeled/poorly labeled dataframe
            if 'lat' not in cols or 'long' not in cols or 'diameter' not in cols:
                if verbose:
                    print('Warning: crater annotations not properly labeled. '
                          'If there is a problem, please reorder crater annotations '
                          'as: x position, y position, crater diameter (pixels) '
                          'and label columns in pandas dataframe.')
                craters.columns = ['lat', 'long', 'diameter']
        else:
            raise Exception('Sorry, labeled craters datatype is not understood. '
                            'Please populate the prediction object '
                            '.known_craters attribute with known crater locations '
                            "(pandas dataframe with columns 'lat', 'long', 'diameter')")
        self.prediction_object = prediction
        self.predicted, self.known = self._match_predictions(
                                        prediction.proposals,
                                        craters,
                                        verbose=verbose
        )
        if verbose:
            print('Matching complete!\n')
        self.done = True
        self.compute_results()
        if verbose:
            self.print_report()
        return

    def plot_densities(self, verbose=True):
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
        if verbose:
            print('known crater count in image: ', len(self.known))
            print('detected crater count in image: ', len(predictions))
        return
        
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
            if not crater.detected:
                y = crater[0]
                x = crater[1]
                r = crater[2]/2
                circle = plt.Circle((x, y), r, fill=False, color='red');
                ax.add_artist(circle);
        for i, proposal in predictions.iterrows():
            if not proposal.positive:
                y = proposal[0]
                x = proposal[1]
                r = proposal[2]/2
                circle = plt.Circle((x, y), r, fill=False, color='yellow');
                ax.add_artist(circle);
            elif proposal.positive:
                y = proposal[0]
                x = proposal[1]
                r = proposal[2]/2
                circle = plt.Circle((x, y), r, fill=False, color='green');
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
    
    def return_stats(self):
        """Returns matched lists of known craters and detections."""
        stats_dict = {
            'true_positives': self.tp,
            'false_positives': self.fp,
            'false_negatives': self.fn,
            'detection_percentage': self.D,
            'precision': self.P,
            'recall': self.R,
            'f1_score': self.F1,
            'false_detections': self.DF,
            'false_negative_rate': self.FNR
        }
        return stats_dict
        
            
        

        
