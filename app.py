from models import (
    BackgroundDetection,
    MotionDetection,
    HeadDetection
)
from player import VideoPlayer

if __name__ == '__main__':
    player = VideoPlayer('Window', 'video/001.mp4')
    player.use(MotionDetection(hide=False, path='bg.jpg'))
    player.use(BackgroundDetection(hide=False))
    player.use(HeadDetection('snapshots/resnet50_v2.h5'))
    # player.register(('b', lambda player, frame: background.save('bg.jpg')))
    player.play()
