from models import (
    BackgroundDetection,
    MotionDetection
)
from player import VideoPlayer

if __name__ == '__main__':
    background = BackgroundDetection()
    motion = MotionDetection()
    player = VideoPlayer('Window', 'video/001.mp4')
    player.register(background)
    player.register(motion)
    player.play()
