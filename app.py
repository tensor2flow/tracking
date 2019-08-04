from models import (
    BackgroundDetection,
    MotionDetection,
    HeadDetection,
    Tracker
)
from player import VideoPlayer

if __name__ == '__main__': 
    player = VideoPlayer('Window', 'video/001.mp4', (600, 1700))
    background = BackgroundDetection(hide=True)
    player.use(MotionDetection(hide=True))
    player.use(background)
    player.use(HeadDetection((600, 700), (1700, 800), 'snapshots/resnet50_v2.h5', hide=False, predict=True))
    player.use(Tracker())
    player.register(('b', lambda player, frame: background.save('test.jpg')))
    player.play()
