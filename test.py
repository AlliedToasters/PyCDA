import unittest
import pycda
from pycda import error_stats as es
from pycda import predictions as pr
from pycda.detectors import _DummyDetector
from pycda.extractors import _DummyExtractor
from pycda.classifiers import _DummyClassifier, ConvolutionalClassifier
from pycda.sample_data import get_sample_image, get_sample_csv
from pycda.util_functions import get_steps, crop_array, make_batch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class TestUtilFuncs(unittest.TestCase):
    
    def setUp(self):
        return
    
    def test_get_steps(self):
        steps_in, steps_out = get_steps(100, 160, 120)
        assert len(steps_in) == len(steps_out)
        assert len(steps_in) == 1
        steps_in, steps_out = get_steps(130, 160, 120)
        assert len(steps_in) == len(steps_out)
        assert len(steps_in) == 2
        steps_in, steps_out = get_steps(241, 160, 120)
        assert len(steps_in) == len(steps_out)
        assert len(steps_in) == 3
    
    def test_crop_array(self):
        test_image = np.random.rand(150, 200)
        test1 = crop_array(test_image, 100)
        assert np.array_equal(test1, test_image[:100, :100])
        test2 = crop_array(test_image, 100, orgn=(-26, -10))
        assert np.array_equal(test2[:26, :10], np.zeros((26, 10)))
        assert np.array_equal(test2[26:, 10:], test_image[:74, :90])
        test3 = crop_array(test_image, 100, orgn=(100, 100))
        assert np.array_equal(test3[50:, :], np.zeros((50, 100)))
        assert np.array_equal(test3[:50, :], test_image[100:, 100:])
        test4 = crop_array(test_image, 100, 20, orgn=(140, 190))
        assert np.array_equal(test4[10:, 10:], np.zeros((90, 10)))
        assert np.array_equal(test4[:10, :10], test_image[140:, 190:])
        
    def test_make_batch(self):
        img_height = np.random.randint(200, 1000)
        img_width = np.random.randint(200, 1000)
        test_image = np.random.rand(img_height, img_width)
        shape = test_image.shape
        height, width = shape[0], shape[1]
        in1 = np.random.randint(2, height)
        in2 = np.random.randint(2, width)
        crops = [
            (0, 0),
            (1, 1),
            (2, 2)
        ]
        try:
            batch = make_batch(test_image, (in1, in2), crops)
        except:
            print('problem batch out dimensions: ', in1, in2)
            raise Exception('Problem building batch.')
        assert batch.shape == (3, in1, in2, 1)

class TestImageFlow(unittest.TestCase):
    
    def setUp(self):
        print('\n')
        self.cda = pycda.CDA(
            detector=_DummyDetector(), 
            extractor=_DummyExtractor(), 
            classifier=_DummyClassifier()
        )
        img_height = np.random.randint(200, 1000)
        img_width = np.random.randint(200, 1000)
        self.test_image = np.random.rand(img_height, img_width)
        self.prediction = pr.Prediction(self.test_image, 'test1', self.cda)
        self.cda.predictions.append(self.prediction)
        
    def test_get_prediction(self):
        prediction = self.cda._get_prediction(self.test_image)
        assert prediction == self.prediction
        img_height = np.random.randint(2, 1000)
        img_width = np.random.randint(2, 1000)
        new_test_image = np.random.rand(img_height, img_width)
        new_prediction = self.cda._get_prediction(new_test_image)
        assert new_prediction != self.prediction
    
    def test_split_image(self):
        self.prediction = self.cda._prepare_detector(self.prediction)
        try:
            assert len(self.prediction.image_split_coords) > 0
            assert len(self.prediction.det_split_coords) > 0
            if self.test_image.shape[0] > self.cda.detector.output_dims[0]:
                assert self.prediction.det_split_coords[-1][0] + self.cda.detector.output_dims[0] \
                == self.test_image.shape[0]
            else:
                assert self.prediction.det_split_coords[-1][0] == 0
            if self.test_image.shape[1] > self.cda.detector.output_dims[1]:
                assert self.prediction.det_split_coords[-1][1] + self.cda.detector.output_dims[1] \
                == self.test_image.shape[1]
            else:
                assert self.prediction.det_split_coords[-1][1] == 0
        except AssertionError:
            print('input img dims: ', self.test_image.shape)
            raise AssertionError()
        
    def test_batch_detect(self):
        batch_size = np.random.randint(1, 5)
        prediction = self.cda.predictions[0]
        prediction = self.cda._prepare_detector(prediction)
        self.cda._batch_detect(prediction, batch_size)
        plt.imshow(self.test_image)
        assert self.test_image.shape ==  prediction.detection_map.shape
        assert np.array_equal(self.test_image, prediction.detection_map)
        
    def test_batch_classify(self):
        batch_size = np.random.randint(1, 100)
        prediction = self.cda.predictions[0]
        prediction.proposals = get_sample_csv()
        prediction.input_image = np.array(get_sample_image().image)
        self.cda._batch_classify(prediction)
        
class TestDetector(unittest.TestCase):
    
    def setUp(self):
        in0 = np.random.randint(150, 250)
        in1 = np.random.randint(150, 250)
        out0 = np.random.randint(50, 150)
        out1 = np.random.randint(50, 150)
        self.detector = _DummyDetector(input_dims=(in0, in1), output_dims=(out0, out1))
    
    def test_dummy_detector(self):
        test_img = np.random.rand(self.detector.input_dims[0], self.detector.input_dims[1])
        batch = np.array([np.expand_dims(test_img, axis=-1)])
        prediction = self.detector.predict(batch)
        offsety = (self.detector.input_dims[0] - self.detector.output_dims[0])//2
        offsetx = (self.detector.input_dims[1] - self.detector.output_dims[1])//2
        yfin = offsety+self.detector.output_dims[0]
        xfin = offsetx+self.detector.output_dims[1]
        assert np.array_equal(prediction[0, :, :, 0], test_img[offsety:yfin, offsetx:xfin])

class TestPrediction(unittest.TestCase):
    
    def setUp(self):
        self.cda = pycda.CDA(
            detector=_DummyDetector(), 
            extractor=_DummyExtractor(), 
            classifier=_DummyClassifier()
        )
        img_height = np.random.randint(500, 1500)
        img_width = np.random.randint(500, 1500)
        self.test_image = np.random.rand(img_height, img_width)
        self.prediction = pr.Prediction(self.test_image, 'test1', self.cda)
        self.cda.predictions.append(self.prediction)
    
    def test_record_detection(self):
        assert self.prediction.detection_map.shape == self.test_image.shape
        ins_y = np.random.randint(5, self.test_image.shape[0])
        ins_x = np.random.randint(5, self.test_image.shape[1])
        self.prediction.det_split_coords.append((ins_y, ins_x))
        try:
            det_y = np.random.randint(5, self.test_image.shape[0]-ins_y)
            det_x = np.random.randint(5, self.test_image.shape[1]-ins_x)
        except ValueError:
            det_y, det_x = 5, 5
        detection = np.random.rand(det_y, det_x)
        self.prediction._record_detection(detection, 0)
        pred_map_slice = self.prediction.detection_map[ins_y:ins_y+det_y, ins_x:ins_x+det_x]
        assert np.array_equal(detection, pred_map_slice)
        
    def test_batch_record_detection(self):
        assert self.prediction.detection_map.shape == self.test_image.shape
        batch = []
        batch_size = np.random.randint(2, 10)
        pred_map_slices = []
        indices = []
        det_y = None
        det_x = None
        for n in range(batch_size):
            indices.append(n)
            if det_y == None:
                ins_y = np.random.randint(5, self.test_image.shape[0])
                ins_x = np.random.randint(5, self.test_image.shape[1])
            else:
                ins_y = np.random.randint(5, self.test_image.shape[0]-det_y)
                ins_x = np.random.randint(5, self.test_image.shape[1]-det_x)
            self.prediction.det_split_coords.append((ins_y, ins_x))
            if det_y == None:
                try:
                    det_y = np.random.randint(5, self.test_image.shape[0]-ins_y)
                    det_x = np.random.randint(5, self.test_image.shape[1]-ins_x)
                except ValueError:
                    det_y = self.test_image.shape[0]-ins_y
                    det_c = self.test_image.shape[1]-ins_x
            detection = np.expand_dims(np.random.rand(det_y, det_x), axis=-1)
            batch.append(detection)
        batch = np.array(batch)
        try:
            self.prediction._batch_record_detection(batch, indices)
        except:
            raise Exception('Error calling ._batch_record_detection')
        
if __name__ in "__main__":
    unittest.main()
