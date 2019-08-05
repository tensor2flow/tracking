from models import (
    BackgroundDetection,
    MotionDetection,
    HeadDetection,
    Tracker
)
from player import VideoPlayer

if __name__ == '__main__': 
    player = VideoPlayer('Window', 'video/003.mp4', (600, 1700), (0, 600), (1100, 700), (0, 400, 1100, 100))
    background = BackgroundDetection(hide=True)
    player.use(MotionDetection(hide=True))
    player.use(background)
    player.use(HeadDetection('snapshots/resnet50_v2.h5', hide=False, predict=True))
    player.use(Tracker())
    player.register(('b', lambda player, frame: background.save('test.jpg')))
    player.play()
