from models import (
    BackgroundDetection,
    MotionDetection
)
from player import VideoPlayer

if __name__ == '__main__':
    background = BackgroundDetection(hide=False)
    motion = MotionDetection(hide=False, path='bg.jpg')
    player = VideoPlayer('Window', 'video/001.mp4')
    player.use(motion)
    player.use(background)
    # player.register(('b', lambda player, frame: background.save('bg.jpg')))
    player.play()
