import cv2 as cv

class MotionDetection:
    def __init__(self, path = None, hide = True):
        self.hide = hide
        self.path = path
        self.background = None
        if self.path is not None:
            self.background = cv.imread(path)
            self.background = cv.cvtColor(self.background, cv.COLOR_BGR2GRAY)

    def run(self, player, frame, threshold = 20):
        if self.path is not None:
            foreground = cv.cvtColor(frame.copy(), cv.COLOR_BGR2GRAY)
            foreground = cv.absdiff(self.background, foreground)
            foreground = cv.threshold(foreground, threshold, 255, cv.THRESH_BINARY)[1]
            countours, _ = cv.findContours(foreground, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
            for countour in countours:
                if 100 <= cv.contourArea(countour):
                    cv.drawContours(frame, [countour], -1, (255, 255, 255), 30)

            if self.hide == False:
                player.windows['Motion'] = foreground

