import torch
import torch.nn as nn
import torch.nn.functional as F

class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels, anchor_ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
        super(RegionProposalNetwork, self).__init__()

        self.anchor_ratios = anchor_ratios
        self.anchor_scales = anchor_scales
        self.num_anchors = len(anchor_ratios) * len(anchor_scales)

        # Convolutional layer for generating anchor scores
        self.conv = nn.Conv2d(in_channels, self.num_anchors * 2, kernel_size=3, padding=1)

        # Convolutional layer for generating anchor regression values
        self.reg_conv = nn.Conv2d(in_channels, self.num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        # Convolutional layer for anchor scores
        scores = self.conv(x)

        # Reshape the scores to have 2 channels per anchor
        scores = scores.view(x.size(0), 2, self.num_anchors, x.size(2), x.size(3))

        # Apply softmax to get anchor scores
        scores = F.softmax(scores, dim=1)

        # Convolutional layer for anchor regression
        regression = self.reg_conv(x)

        # Reshape the regression values to have 4 values per anchor
        regression = regression.view(x.size(0), 4, self.num_anchors, x.size(2), x.size(3))

        return scores, regression

# Example usage
in_channels = 256  # Number of channels from the backbone network
rpn = RegionProposalNetwork(in_channels)

# Input tensor (batch size, channels, height, width)
x = torch.randn(1, in_channels, 50, 50)

# Forward pass
scores, regression = rpn(x)

# Print the shapes of the output
print("Anchor Scores Shape:", scores.shape)
print("Anchor Regression Shape:", regression.shape)
