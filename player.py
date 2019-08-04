import cv2 as cv
import numpy as np
import imutils
from time import time

class VideoPlayer:
    def __init__(self, name, path, clipped_size):
        self.clipped_size = clipped_size
        self.name = name
        self.orginal = None
        self.video = cv.VideoCapture(path)
        self.models = []
        self.windows = {}
        self.frame = None
        self.scale = 1.0
        self.stopped = False
        self.boxes = None
        self.predictions = None
        self.events = [
            ( 'q', lambda player, frame: player.stop() )
        ]

    def stop(self):
        self.stopped = True

    def use(self, model):
        self.models.append(model)
    
    def register(self, event):
        self.events.append(event)

    def play(self):
        while True:
            if self.stopped:
                break
            check, frame = self.video.read()
            if frame is None:
                break
            start = time()
            #frame = frame[0:frame.shape[0], self.clipped_size[0]:self.clipped_size[1]]
            self.orginal = frame.copy()
            width = frame.shape[1]
            frame = imutils.resize(frame, width=900)
            self.scale = width / 900
            for model in self.models:
                model.run(self, frame)
            cv.imshow(self.name, self.orginal)
            for name in self.windows:
                cv.imshow(name, self.windows[name])
            key = cv.waitKey(1)
            for code, action in self.events:
                if key == ord(code):
                    action(self, frame)
            print('performance : {}'.format(time() - start))
        cv.destroyAllWindows()
        self.video.release()