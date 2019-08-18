import cv2 as cv
import numpy as np
from time import time
import settings

class Tracker:
    def __init__(self, frame, predictions):
        self.trackers = {}
        self.lastID = 0
        self.predictions = {}
        for box in predictions:
            self.trackers[self.lastID] = settings.Tracker()
            self.trackers[self.lastID].init(frame, (box[0], box[1], box[2] - box[0], box[3] - box[1]))
            self.lastID += 1

    def track(self, frame):
        rects = []
        for uuid in self.trackers.keys():
            success, box = self.trackers[uuid].update(frame)
            box = np.int0(box)
            if success:
                #cv.rectangle(player.orginal, (box[0], box[1], box[2], box[3]), (255, 0, 0), 2)
                rects.append((box[0], box[1], box[2] + box[0], box[3] + box[1]))
        return rects