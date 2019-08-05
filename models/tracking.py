import cv2 as cv
import numpy as np
from time import time
from scipy.spatial import distance as dist
from collections import OrderedDict

class CentroidTracker():
	def __init__(self):
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
		self.news, self.all = {}, {}
		self.removes = {}

	def register(self, centroid, rect):
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.news[self.nextObjectID] = rect; self.all[self.nextObjectID] = rect
		self.nextObjectID += 1

	def deregister(self, objectID):
		self.removes[objectID] = True; del self.all[objectID]
		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, rects):
		self.news = {}; self.removes = {}
		if len(rects) == 0:
			disappeared = self.disappeared.copy()
			for objectID in disappeared.keys():
				self.deregister(objectID)
				
			return self.objects

		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i], rects[i])
		else:
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			D = dist.cdist(np.array(objectCentroids), inputCentroids)

			rows = D.min(axis=1).argsort()

			cols = D.argmin(axis=1)[rows]
			usedRows = set()
			usedCols = set()

			for (row, col) in zip(rows, cols):
				if row in usedRows or col in usedCols:
					continue
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0

				usedRows.add(row)
				usedCols.add(col)

			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            
			if D.shape[0] >= D.shape[1]:
				for row in unusedRows:
					objectID = objectIDs[row]
					self.deregister(objectID)

			else:
				for col in unusedCols:
					self.register(inputCentroids[col], rects[col])

		return self.objects

class Tracker:
    def __init__(self):
        self.trackers = {}
        self.ct = CentroidTracker()
        self.results = {}
        self.lastID = 1
        self._in = 0
        self._out = 0

    def run(self, player, _):
        rects = []
        if player.predictions is not None:
            self.trackers = {}
            for box in player.predictions:
                self.trackers[self.lastID] = cv.TrackerCSRT_create()
                self.trackers[self.lastID].init(player.orginal, (box[0], box[1], box[2] - box[0], box[3] - box[1]))
                self.lastID += 1
            for box in player.predictions:
                rects.append((box[0], box[1], box[2], box[3]))
                cv.rectangle(player.orginal, (box[0], box[1], box[2] - box[0], box[3] - box[1]), (0, 255, 0), 2)
        else:
            for uuid in self.trackers.keys():
                success, box = self.trackers[uuid].update(player.orginal)
                box = np.int0(box)
                if success:
                    cv.rectangle(player.orginal, (box[0], box[1], box[2], box[3]), (255, 0, 0), 2)
                    rects.append((box[0], box[1], box[2] + box[0], box[3] + box[1]))
        self.ct.update(rects)
        for uuid in self.ct.objects.keys():
            point = self.ct.objects[uuid]
            if uuid not in self.results.keys():
                _type = 0
                if player.area[1] <= point[1] <= player.area[1] + player.area[3]:
                    _type = 0
                if player.area[1] > point[1]:
                    _type = -1
                if player.area[1] + player.area[3] < point[1]:
                    _type = 1
                self.results[uuid] = {
                    'type': _type
                }
            else:
                if player.area[1] <= point[1] <= player.area[1] + player.area[3]:
                    if self.results[uuid]['type'] == -1:
                        self.results[uuid]['type'] = 0
                        self._in += 1
                    if self.results[uuid]['type'] == 1:
                        self.results[uuid]['type'] = 0
                        self._out += 1
                else:
                    if self.results[uuid]['type'] == 0:
                        if player.area[1] <= point[1] <= player.area[1] + player.area[3]:
                            _type = 0
                        if player.area[1] > point[1]:
                            _type = -1
                        if player.area[1] + player.area[3] < point[1]:
                            _type = 1
                        self.results[uuid]['type'] = _type

            _type = 'OUT' if self.results[uuid]['type'] == -1 else (
                'IN' if self.results[uuid]['type'] == 1 else 'SCN'
            )
            cv.putText(
                player.orginal, 
                'ID: {}, {}'.format(uuid, _type), 
                (point[0], point[1]), 
                cv.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                2
            )
            cv.circle(player.orginal, (point[0], point[1]), 4, (0, 255, 0), -1)
        cv.rectangle(player.orginal, player.area, (0, 255, 0),2)
        cv.putText(player.orginal, 'IN : {}'.format(self._in), (100, 100), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #cv.putText(player.orginal, 'OUT : {}'.format(self._out), (100, 130), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        player.predictions = None