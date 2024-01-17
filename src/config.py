import torch
import os

# increase/ decrease according to GPU memory - Number of samples in each batch during training.
BATCH_SIZE = 16
# Resize the images during training and transformation to this size.
RESIZE_TD = [640, 640]
NUM_EPOCHS = 500  # Number of epochs to train for.

print("--- Check Device run code ---")
DEVICE = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")
print("DEVICE:", DEVICE)
print("---||---||---")

# Path to the directory containing annotations.
ANNOTS_DIR = "./dataset/test_data/50_xml"
# Path to the directory containing images.
IMAGES_DIR = "./dataset/test_data/50_image"

if not os.path.isdir(ANNOTS_DIR):
    print(f"Label train data is not exit: {ANNOTS_DIR}")
    exit()
if not os.path.isdir(IMAGES_DIR):
    print(f"Image train data is not exit: {IMAGES_DIR}")
    exit()

# Ratio for splitting the dataset into training and validation sets.
SPLIT_RATIO = 0.3
# classes: 0 index is reserved for background
CLASSES = ["background", "3", "4", "5"]
NUM_CLASSES = 4

# Whether to visualize images after creating the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = False

# Directory to save the model and plots
OUT_DIR = "./outputs_model"

# Tuple containing allowed image file extensions.
IMAGE_TYPE = (".jpg", ".png", ".bmp")

if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)

# Save loss plots after a certain number of epochs.
SAVE_PLOTS_EPOCH = 5
# Save the model after a certain number of epochs.s
SAVE_MODEL_EPOCH = 5
