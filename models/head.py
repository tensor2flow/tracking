from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.models.retinanet import retinanet_bbox
import numpy as np
from keras_retinanet.utils.visualization import draw_box, draw_caption
import cv2 as cv
from keras import backend as K
from time import time

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
        if player.i % 10 == 0:
            isprocessing = False
            if player.boxes is not None:
                for box in player.boxes:
                    x1, y1, x2, y2 = box
                    x1, y1, x2, y2 = player.scale * x1, player.scale * y1, player.scale * x2, player.scale * y2
                    if player.a[0] < x1 < player.b[0] and player.a[0] < x2 < player.b[0] and player.a[1] > y1 and y2 > player.b[1]:
                        isprocessing = True
            if isprocessing and self.ispredict or self.ispredict and player.last_predicted + 30 < player.i:
                start = time()
                boxes = self.predict(player.orginal)
                player.last_predicted = player.i
                #print('performance:', time() - start)
                player.predictions = boxes
        cv.rectangle(player.orginal, player.a, player.b, (255, 0, 0), 1)