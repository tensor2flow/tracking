import cv2 as cv

class VideoPlayer:
    def __init__(self, name, path):
        self.name = name
        self.video = cv.VideoCapture(path)
        self.models = []
        self.windows = {}
        self.stopped = False
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
            for model in self.models:
                model.run(self, frame)
            cv.imshow(self.name, frame)
            for name in self.windows:
                cv.imshow(name, self.windows[name])
            key = cv.waitKey(1)
            for code, action in self.events:
                if key == ord(code):
                    action(self, frame)
        cv.destroyAllWindows()
        self.video.release()