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
    def __init__(self, path, baskbone='resnet50', hide=True, predict=True):
        self.ispredict = predict
        self.hide = hide
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

    def run(self, player, frame):
        if player.i % 25 == 0:
            start = time()
            boxes = self.predict(player.orginal)
            player.last_predicted = player.i
            #print('performance:', time() - start)
            player.predictions = boxes
        cv.rectangle(player.orginal, player.a, player.b, (255, 0, 0), 1)