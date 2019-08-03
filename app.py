import cv2 as cv
import numpy as np

if __name__ == '__main__':
    background = BackgroundDetection()
    motion = MotionDetection()
    player = VideoPlayer('Window', '002.mp4')
    player.register(background)
    player.register(motion)
    player.play()
