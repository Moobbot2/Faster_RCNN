import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from module.ResNet51 import ResNet51, Bottleneck

# Assume resnet51 is your ResNet-51 model
resnet51 = ResNet51(Bottleneck, [3, 4, 6, 3])

# Pre-trained Faster R-CNN with resnet50_fpn as backbone
model_frcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn()

# Replace the backbone of Faster R-CNN with your ResNet-51
model_frcnn.backbone.body = resnet51

# Modify the number of classes in the final classification layer of the Faster R-CNN model
num_classes = 20  # Change this based on your task
in_features = model_frcnn.roi_heads.box_predictor.cls_score.in_features
model_frcnn.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Optional: Modify other components of the Faster R-CNN model if needed

# Now, model_frcnn is your Faster R-CNN model with ResNet-51 backbone
