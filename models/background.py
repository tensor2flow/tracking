import cv2 as cv

class BackgroundDetection:
    def __init__(self, history = 5000, nmixtures=4, backgroundRatio=0.0001, **kwargs):
        self.background = None
        self.model = cv.bgsegm.createBackgroundSubtractorMOG(
            history=5000,
            nmixtures=5,
            backgroundRatio=0.0001,
            **kwargs
        )
    
    def run(self, player, frame):        
        pass