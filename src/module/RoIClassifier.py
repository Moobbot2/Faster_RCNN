import torch
import torch.nn as nn

class RoIClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(RoIClassifier, self).__init__()
        self.fc1 = nn.Linear(in_channels * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class RoIRegressor(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super(RoIRegressor, self).__init__()
        self.fc1 = nn.Linear(in_channels * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, num_anchors * 4)  # 4 for (dx, dy, dw, dh)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Example usage
in_channels = 256  # This should match the output channels from the RoI pooling layer
num_classes = 20  # Adjust based on the number of classes in your dataset
num_anchors = 9  # Adjust based on the number of anchors used in the RPN

roi_classifier = RoIClassifier(in_channels, num_classes)
roi_regressor = RoIRegressor(in_channels, num_anchors)

# Example RoI-pooled features
roi_pooled_features = torch.randn(200, in_channels, 7, 7)  # Example for 200 RoIs

# Forward pass through classifier
class_scores = roi_classifier(roi_pooled_features)

# Forward pass through regressor
bbox_deltas = roi_regressor(roi_pooled_features)

# Print the shapes of the outputs
print("Class Scores Shape:", class_scores.shape)
print("Bbox Deltas Shape:", bbox_deltas.shape)
