import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.models as models


def create_model(num_classes):

    # load Faster RCNN pre.trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

    # get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the deteter with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
