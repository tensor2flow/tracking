from player import VideoPlayer

if __name__ == '__main__': 
    player = VideoPlayer('Window', 'video/003.mp4', (600, 1700),  320)
    player.play()