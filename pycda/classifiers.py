import numpy as np

class ClassifierBaseClass(object):
    """Base object for crater classifier. Classifiers
    make a binary prediction on a crater proposal and return
    a value between zero and one; one represents a true crater
    and zero represents a false proposal.
    """
    def __init__(self):
        #Specify input dimensions for crater classifier
        self.input_dims = (0, 0)
        #specify the diameter of the crater candidate
        #should have in the model input, in pixels
        self.crater_pixels = 0
        #Specify number of input color channels
        self.input_channels = 1
        #Recommend batch size reasonable for modest computer
        self.rec_batch_size = 32
        
    def predict(self, batch):
        """Prediction call should return an array of predictions
        of length of batch.
        """
        raise Exception('Base classifier cannot make predictions.')
        
class DummyClassifier(ClassifierBaseClass):
    """Dummy classifier for testing."""
    
    def __init__(self, input_dims=(20, 20), n_channels=1):
        self.input_dims = input_dims
        self.crater_pixels = 8
        self.input_channels = n_channels
        self.rec_batch_size = 32
        
    def predict(self, batch):
        """Returns an array of randomly-generated predictions of length
        of batch."""
        try:
            assert (batch.shape[1], batch.shape[2]) == self.input_dims
        except AssertionError:
            raise Exception('input image shape must match classifier.input_dims')
        batch_size = batch.shape[0]
        predictions = []
        for i in range(batch_size):
            prediction = np.random.rand()
            predictions.append(prediction)
        return np.array(predictions)
    
class NullClassifier(ClassifierBaseClass):
    """For use when classifier is not wanted. Returns a likelihood of 1
    for every proposal."""
    
    def __init__(self, input_dims = (1, 1)):
        self.input_dims = input_dims
        self.crater_pixels = 1
        self.rec_batch_size = 100
        
    def predict(self, batch):
        """Returns an array of randomly-generated predictions of length
        of batch."""
        batch_size = batch.shape[0]
        predictions = [1 for x in range(batch_size)]
        return np.array(predictions)
        
class ConvolutionalClassifier(ClassifierBaseClass):
    """12x12 pixel classifier using 2D convolution
    implimented with Keras on tensorflow backend. Built
    for nice performance and speed."""
    
    def __init__(self):
        import tensorflow as tf
        from keras.models import load_model
        self.model = load_model('./models/classifier_12x12_2.h5')
        self.input_dims = (12, 12)
        self.crater_pixels = 4
        self.input_channels = 1
        self.rec_batch_size = 128
        
    def predict(self, batch):
        """Performs prediction on batch."""
        return self.model.predict(batch)
        
        
def get(identifier):
    """handles argument to CDA pipeline for classifier specification.
    returns an initialized classifier.
    """
    model_dictionary = {
        'convolution': ConvolutionalClassifier,
        'dummy': DummyClassifier,
        'none': NullClassifier
    }
    if identifier is None:
        return NullClassifier()
    if isinstance(identifier, ClassifierBaseClass):
        model = identifier
        return model
    elif identifier in model_dictionary:
        return model_dictionary[identifier]()
    elif callable(identifier):
        if isinstance(identifier(), ClassifierBaseClass):
            return identifier()
        else:
            raise Exception('custom classifiers must inherit'
                           'from ClassifierBaseClass, which can be'
                           'imported from classifiers.py')
        return identifier()
    else:
        raise ValueError('Could not interpret '
                         'classifier identifier: {} \n'
                         'try one of these keywords: {}'
                         .format(identifier, list(model_dictionary.keys())))
        
        