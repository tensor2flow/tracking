from imagenet import models
from imagenet.utils.image import preprocess_image, resize_image
from imagenet.models.retinanet import retinanet_bbox
from time import time
import numpy as np
from imagenet.utils.visualization import draw_box, draw_caption
import cv2 as cv
from keras import backend as K

K.clear_session()

class HeadDetection:
    def __init__(self, a, b, path, baskbone='resnet50', hide=True):
        self.a, self.b = a, b
        self.hide = hide
        self.train_model = models.load_model(path, backbone_name=baskbone)
        self.prediction_model = retinanet_bbox(self.train_model, anchor_params=None)
    def predict(self, image, **kwargs):
        image = preprocess_image(image.copy())
        image, scale = resize_image(image, 500, 800, **kwargs)
        cv.imshow('Example', image)
        boxes, scores, _ = self.prediction_model.predict_on_batch(np.expand_dims(image, axis=0))
        #boxes, scores, _ = self.prediction_model.predict_on_batch(image)
        boxes /= scale
        results = []
        for box, score in zip(boxes[0], scores[0]):
            if score < 5.0:
                break
            results.append(np.int0(box))
        return np.array(results)

    def run(self, player, frame):
        cv.rectangle(player.orginal, self.a, self.b, (255, 0, 0), 1)
        if self.hide == False:
            isprocessing = False
            if player.boxes is not None:
                for box in player.boxes:
                    x1, y1, x2, y2 = box
                    if self.a[0] < x1 < self.b[0] and self.a[0] < x2 < self.b[0] and self.a[1] > y1 and y2 > self.b[1]:
                        isprocessing = True
            if isprocessing:
                #draw_box(player.orginal, (x1, y1, x2, y2), color, 1)
                #draw_caption(player.orginal, (x1, y1, x2, y2), caption)
                start = time()
                boxes = self.predict(player.orginal)
                print(boxes)
                for box in boxes:
                    print(box)
                print('performance : {}'.format(time() - start))