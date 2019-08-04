import cv2 as cv
import numpy as np
from imagenet.utils.visualization import draw_box, draw_caption

class BackgroundDetection:
    def __init__(self, history = 10000, nmixtures=4, backgroundRatio=0.0001, hide=True, **kwargs):
        self.active = None
        self.last = None
        self.min = None
        self.hide = hide
        self.model = cv.bgsegm.createBackgroundSubtractorMOG(
            history=5000,
            nmixtures=5,
            backgroundRatio=0.0001,
            **kwargs
        )

    def save(self, path):
        cv.imwrite(path, self.last)

    def run(self, player, frame):
        self.last = frame.copy()
        self.logic = self.model.apply(frame)
        
        if self.min is None:
            self.min = self.logic.shape[0] * self.logic.shape[1]
            self.active = frame.copy()
        else:
            mn = np.count_nonzero(self.logic)
            if self.min > mn:
                self.min = mn
                self.active = frame.copy()
        countours, check = cv.findContours(self.logic, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
        player.boxes = None
        for i, countour in enumerate(countours):
            if 1000 <= cv.contourArea(countour):
                x, y, w, h = cv.boundingRect(countour)
                #cv.drawContours(frame, [countour], -1, (255, 0, 0), 3)
                #cv.rectangle(player.orginal, (x, y), (x + w, y + h), (0, 255, 0), 4)
                if check[0, i, 3] == -1:
                    box = np.array([[x, y, x + w, y + h]])
                    if player.boxes is None:
                        player.boxes = box
                    else:
                        player.boxes = np.append(player.boxes, box, axis=0)
        if self.hide == False:
            player.windows['Background'] = self.logic