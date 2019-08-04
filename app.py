from models import (
    BackgroundDetection,
    MotionDetection,
    HeadDetection
)
from player import VideoPlayer

if __name__ == '__main__': 
    player = VideoPlayer('Window', 'video/001.mp4')
    player.use(MotionDetection(hide=True, path='bg.jpg'))
    player.use(BackgroundDetection(hide=True))
    player.use(HeadDetection((180, 200), (580, 250), 'snapshots/resnet50.h5', hide=False))
    # player.register(('b', lambda player, frame: background.save('bg.jpg')))
    player.play()
