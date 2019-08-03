import cv2 as cv

class VideoPlayer:
    def __init__(self, name, path):
        self.name = name
        self.video = cv.VideoCapture(path)
        self.listeners = []
        self.stop = False
        def quit(player, frame):
            player.stop = True
        self.events = [
            ( 'q', quit )
        ]

    def register(self, listener):
        self.listeners.append(listener)

    def play(self):
        while True:
            if self.stop:
                break
            check, frame = self.video.read()
            if frame is None:
                break
            for listener in self.listeners:
                listener.run(self, frame)
            cv.imshow(self.name, frame)
            key = cv.waitKey(1)
            for code, action in self.events:
                if key == ord(code):
                    action(self, frame)
        cv.destroyAllWindows()
        self.video.release()