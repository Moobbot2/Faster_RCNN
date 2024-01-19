import ResNet51 as ResNet51


# Create an instance of ResNet-50
resnet51 = ResNet51.ResNet51(ResNet51.Bottleneck, [3, 4, 6, 3])

# Print the model architecture
print(resnet51)