import cv2 as cv
import numpy as np

class Tracker:
    def __init__(self):
        self.trackers = {}
        self.lastID = 1
    
    def run(self, player, _):
        if player.predictions is not None:
            self.trackers = {}
            for box in player.predictions:
                box = np.int0(box)
                self.trackers[self.lastID] = cv.TrackerCSRT_create()
                self.trackers[self.lastID].init(player.orginal, (box[0], box[1], box[2] - box[0], box[3] - box[1]))
                
        for objectId in self.trackers:
            success, box = self.trackers[objectId].update(player.orginal)
            if success:
                (x, y, w, h) = np.int0(box)
                cv.rectangle(player.orginal, (x, y), (x + w, y + h), (0, 255, 0), 2)
        player.predictions = None