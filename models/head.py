from imagenet import models
from imagenet.utils.image import preprocess_image, resize_image
from imagenet.models.retinanet import retinanet_bbox

class HeadDetection:
    def __init__(self, path, baskbone='resnet50'):
        self.train_model = models.load_model(path, backbone_name=baskbone)
        self.prediction_model = retinanet_bbox(self.train_model, anchor_params=None)
    
    def predict(self, image, **kwargs):
        image = preprocess_image(image.copy())
        image, scale = resize_image(image, **kwargs)
        boxes, scores, _ = self.prediction_model.predict_on_batch(np.expand_dims(image, axis=0))
        results = []
        for box, score in zip(boxes[0], scores[0]):
            if score < 5.0:
                break
            results.append(np.int0(box))
        return np.array(results)

    def run(self, player, frame):
        pass