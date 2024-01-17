import torch
import torch.nn as nn
import torch.nn.functional as F


class RoIPooling(nn.Module):
    def __init__(self, output_size):
        super(RoIPooling, self).__init__()
        self.output_size = output_size

    def forward(self, features, rois):
        """
        RoI pooling forward pass.

        Parameters:
        - features: Feature map from the backbone network (Tensor of shape [N, C, H, W])
        - rois: Region of Interest coordinates (Tensor of shape [num_rois, 4], where each row is [x1, y1, x2, y2])

        Returns:
        - Pooled features for each RoI (Tensor of shape [num_rois, C, output_size, output_size])
        """
        num_rois = rois.size(0)
        output_size = self.output_size

        # Initialize an empty tensor to store the pooled features
        pooled_features = torch.zeros(num_rois, features.size(
            1), output_size, output_size, requires_grad=True)

        for i in range(num_rois):
            roi = rois[i]
            x1, y1, x2, y2 = roi

            # Calculate the spatial dimensions of the RoI
            roi_width = max(x2 - x1 + 1, 1)
            roi_height = max(y2 - y1 + 1, 1)

            # Calculate the size of each spatial bin
            bin_size_x = roi_width / output_size
            bin_size_y = roi_height / output_size

            for h in range(output_size):
                for w in range(output_size):
                    # Calculate the coordinates in the original feature map corresponding to the RoI bin
                    bin_x1 = int(x1 + bin_size_x * w)
                    bin_x2 = int(x1 + bin_size_x * (w + 1))
                    bin_y1 = int(y1 + bin_size_y * h)
                    bin_y2 = int(y1 + bin_size_y * (h + 1))

                    # Use RoI bin coordinates to extract the corresponding features and perform max pooling
                    roi_bin = features[:, :, bin_y1:bin_y2, bin_x1:bin_x2]

                    # Use clone() to avoid in-place operation on a leaf variable
                    pooled_features[i, :, h, w] = F.adaptive_max_pool2d(
                        roi_bin.clone(), 1).view(-1)
                    
        return pooled_features


# Example usage
output_size = 7  # Adjust this based on the desired output size
roi_pooling_layer = RoIPooling(output_size)

# Example feature map from the backbone network
features = torch.randn(1, 256, 50, 50)

# Example RoI coordinates (x1, y1, x2, y2) for one RoI
rois = torch.tensor([[10, 15, 35, 45]])

# Forward pass
pooled_features = roi_pooling_layer(features, rois)

# Print the shape of the output
print("Pooled Features Shape:", pooled_features.shape)
