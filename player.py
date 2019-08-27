import cv2 as cv
import numpy as np
import imutils
from time import time
from imagenet.utils.image import preprocess_image, resize_image
from tqdm import tqdm
from models import (
    HeadDetection,
    Tracker,
    CentroidTracker
)

def getState(x, y, line):
    if y > line:
        return -1
    else:
        return 1

def getReports(objects):
    results = 0
    for uuid in objects.keys():
        positions = objects[uuid]
        if positions['last'] == -1 and positions['seen'] == 1:
            results += 1
    return results

class VideoPlayer:
    def __init__(self, name, path, clipped_size, line):
        self.clipped_size = clipped_size
        self.line = line
        self.name = name
        self.orginal = None
        self.video = cv.VideoCapture(path)
        self.out = None
        self.model = HeadDetection('snapshots/resnet50_v2.h5')
        self.frame = None
        self.scale = 1.0
        self.stopped = False
        self.boxes = None
        self.predictions = None
        self.events = [
            ( 'q', lambda player, frame: player.stop() )
        ]
        self.i = 0
        self.queries = []
        self.frames = []
        self.results = 0
        self.ct = CentroidTracker()

    def stop(self):
        self.stopped = True
    
    def register(self, event):
        self.events.append(event)

    def run(self):
        limit = 10
        if len(self.queries) > limit * 2 * 30:
            st = time()
            prediction_frames = []
            groups = []
            scale = 1.0
            while len(self.queries) > limit * 2:
                frames , self.queries = self.queries[:limit * 2], self.queries[limit * 2:]
                groups.append(frames)
                image = frames[limit]
                image = preprocess_image(image.copy())
                image, scale = resize_image(image)
                prediction_frames.append(image)
            prediction_boxes = self.model.predicts(prediction_frames, scale)
            for index, frames in enumerate(groups):
                back, prediction_frame, front = frames[:limit], frames[limit], frames[limit:]
                boxes = prediction_boxes[index]
                back_tracker, front_tracker = (
                    Tracker(prediction_frame, boxes), Tracker(prediction_frame, boxes)
                )
                back = np.array(back)
                front = np.array(front)
                predictions = [ None ] * (limit * 2)
                for i in range(back.shape[0] - 1, -1, -1):
                    a = back_tracker.track(back[i])
                    predictions[i] = (back[i], a)
                for i in range(0, front.shape[0]):
                    b = front_tracker.track(front[i])
                    predictions[limit + i] = (front[i], b)
                self.frames.extend(predictions)
            #print('performance:', time() - st)

    def play(self):
        st = time()
        for _ in tqdm(range(int(self.video.get(cv.CAP_PROP_FRAME_COUNT)))):
            if self.stopped:
                break
            check, frame = self.video.read()
            if frame is None:
                break
            start = time()
            frame = frame[0:frame.shape[0], self.clipped_size[0]:self.clipped_size[1]]
            width = frame.shape[1]
            frame = imutils.resize(frame, width=700)
            self.scale = width / 700
            self.queries.append(frame)
            self.run()

            if self.out is None:
                self.out = cv.VideoWriter(
                    'out.avi',
                    cv.VideoWriter_fourcc('M','J','P','G'), 
                    25,
                    (frame.shape[1], frame.shape[0])
                )

            if len(self.frames) > 0:
                (frame, boxes), self.frames = self.frames[0], self.frames[1:]
                reacts = []
                for box in boxes:
                    cv.rectangle(frame, (box[0], box[1], box[2] - box[0], box[3] - box[1]), (255, 0, 0), 2)
                    reacts.append((box[0], box[1], box[2], box[3]))
                objects = self.ct.update(reacts)
                for (objectID, centroid) in objects.items():
                    text = "ID {}".format(objectID)
                    cv.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                    if self.ct.all[objectID]['seen'] is None:
                        self.ct.all[objectID]['seen'] = getState(centroid[0], centroid[1], self.line)
                    self.ct.all[objectID]['last'] = getState(centroid[0], centroid[1], self.line)
                cv.line(frame, (10, self.line), (frame.shape[1] - 10, self.line), (0, 255, 0), 2)
                results = getReports(self.ct.all)
                cv.putText(frame, 'Results : {}'.format(results), (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
                #cv.imshow(self.name, frame)
                self.out.write(frame)

            key = cv.waitKey(1)
            for code, action in self.events:
                if key == ord(code):
                    action(self, frame)
            #print('performance : {}'.format(time() - start))
            self.i += 1

        while len(self.frames) > 0:
            (frame, boxes), self.frames = self.frames[0], self.frames[1:]
            reacts = []
            for box in boxes:
                cv.rectangle(frame, (box[0], box[1], box[2] - box[0], box[3] - box[1]), (255, 0, 0), 2)
                reacts.append((box[0], box[1], box[2], box[3]))
            objects = self.ct.update(reacts)
            for (objectID, centroid) in objects.items():
                text = "ID {}".format(objectID)
                cv.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                if self.ct.all[objectID]['seen'] is None:
                    self.ct.all[objectID]['seen'] = getState(centroid[0], centroid[1], self.line)
                self.ct.all[objectID]['last'] = getState(centroid[0], centroid[1], self.line)
            cv.line(frame, (10, self.line), (frame.shape[1] - 10, self.line), (0, 255, 0), 2)
            results = getReports(self.ct.all)
            cv.putText(frame, 'Results : {}'.format(results), (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))
            #cv.imshow(self.name, frame)
            self.out.write(frame)
        self.video.release()
        self.out.release()
        cv.destroyAllWindows()
        print('performance:', time() - st)