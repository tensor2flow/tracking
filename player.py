import cv2 as cv
import numpy as np

class VideoPlayer:
    def __init__(self, name, path):
        self.name = name
        self.orginal = None
        self.video = cv.VideoCapture(path)
        self.models = []
        self.windows = {}
        self.frame = None
        self.stopped = False
        self.boxes = None
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
            self.orginal = frame.copy()
            if frame is None:
                break
            for model in self.models:
                model.run(self, frame)
            cv.imshow(self.name, self.orginal)
            for name in self.windows:
                cv.imshow(name, self.windows[name])
            key = cv.waitKey(1)
            for code, action in self.events:
                if key == ord(code):
                    action(self, frame)
        cv.destroyAllWindows()
        self.video.release()