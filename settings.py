import cv2 as cv

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv.TrackerCSRT_create,
    "kcf": cv.TrackerKCF_create,
    "boosting": cv.TrackerBoosting_create,
    "mil": cv.TrackerMIL_create,
    "tld": cv.TrackerTLD_create,
    "medianflow": cv.TrackerMedianFlow_create,
    "mosse": cv.TrackerMOSSE_create
}

OPENCV_OBJECT_TRACKER = "csrt"
Tracker = OPENCV_OBJECT_TRACKERS[OPENCV_OBJECT_TRACKER]