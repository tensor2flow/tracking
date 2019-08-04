import cv2 as cv
import numpy as np

from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
	def __init__(self, maxDisappeared=50):
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
		self.news = {}
		self.removes = {}

		self.maxDisappeared = maxDisappeared

	def register(self, centroid, rect):
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.news[self.nextObjectID] = rect
		self.nextObjectID += 1

	def deregister(self, objectID):
		self.removes[objectID] = True
		del self.objects[objectID]
		del self.disappeared[objectID]

	def update(self, rects):
		self.news = {}
		if len(rects) == 0:
			disappeared = self.disappeared.copy()
			for objectID in self.disappeared.keys():
				disappeared[objectID] += 1

			for objectID in disappeared.keys():
				if disappeared[objectID] > self.maxDisappeared:
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
					self.disappeared[objectID] += 1

					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)

			else:
				for col in unusedCols:
					self.register(inputCentroids[col], rects[col])

		return self.objects

class Tracker:
    def __init__(self):
        self.trackers = {}
        self.ct = CentroidTracker()
    
    def run(self, player, _):
        status = -1
        if player.predictions is not None:
            
            rects = []
            for box in player.predictions:
                status = 0
                box = np.int0(box)
                rects.append((box[0], box[1], box[2] - box[0], box[3] - box[1]))
                self.ct.update(rects)
        if status == 0:
            for uuid in self.ct.news.keys():
                self.trackers[uuid] = cv.TrackerCSRT_create()
                self.trackers[uuid].init(player.orginal, self.ct.news[uuid])
            for uuid in self.ct.removes.keys():
                if uuid in self.trackers.keys():
                    del self.trackers[uuid]
    
        for objectId in self.trackers:
            success, box = self.trackers[objectId].update(player.orginal)
            if success:
                (x, y, w, h) = np.int0(box)
                cv.rectangle(player.orginal, (x, y), (x + w, y + h), (0, 255, 0), 2)
        player.predictions = None