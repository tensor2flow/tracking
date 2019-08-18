from imagenet import models
from imagenet.utils.image import preprocess_image, resize_image
from imagenet.models.retinanet import retinanet_bbox
import numpy as np
from imagenet.utils.visualization import draw_box, draw_caption
import cv2 as cv
from keras import backend as K
from time import time
import settings

K.clear_session()

class HeadDetection:
    def __init__(self, path, baskbone='resnet50'):
        self.train_model = models.load_model(path, backbone_name=baskbone)
        self.prediction_model = retinanet_bbox(self.train_model, anchor_params=None)

    def predict(self, image, **kwargs):
        image = preprocess_image(image.copy())
        image, scale = resize_image(image)
        boxes, scores, _ = self.prediction_model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale
        results = []
        for box, score in zip(boxes[0], scores[0]):
            if score < 0.60:
                break
            results.append(np.int0(box))
        return np.array(results)
    
    def predicts(self, images, scale):
        results = []
        boxes, scores, _ = self.prediction_model.predict_on_batch(np.array(images))
        boxes /= scale
        for i in range(len(images)):
            result = []
            for box, score in zip(boxes[i], scores[i]):
                if score < 0.60:
                    break
                result.append(np.int0(box))
            results.append(result)
        return results
