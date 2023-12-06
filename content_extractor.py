import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models.vgg import VGG16_Weights
import matplotlib.pyplot as plt
import numpy as np

# Load the pretrained VGG16 model
model = models.vgg16(weights=VGG16_Weights.DEFAULT).features

# Freeze all the layers of the model
for param in model.parameters():
    param.requires_grad = False

# Set the model to evaluation mode
model.eval()

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def extract_content(image, layer_idx=21):
    # Define the image transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])

    # Load and transform the image
    image = transform(image).unsqueeze(0).to(device)

    # Extract content features
    with torch.no_grad():
        for i, layer in enumerate(model.children()):
            image = layer(image)
            if i == layer_idx:
                break

    return image


def visualize_feature_map(feature_map, map_index=0):
    """
    Visualize a single feature map from a layer output.

    Parameters:
    feature_map (torch.Tensor): The output tensor from a layer of the model.
    map_index (int): Index of the feature map to visualize.
    """
    # Ensure the feature map is detached and moved to CPU
    feature_map = feature_map.detach().cpu()

    # Select the feature map and squeeze out the batch dimension
    single_map = feature_map[0, map_index, :, :]

    # Normalize the feature map
    single_map -= single_map.min()
    single_map /= single_map.max()

    # Plot the feature map
    plt.imshow(single_map)
    plt.colorbar()
    plt.show()

# Example usage
# Assuming 'extracted_features' is the output from your 'extract_content' function
# image = Image.open("dog.jpg").convert("RGB")
# content = extract_content(image)
# visualize_feature_map(content, map_index=0)


# print(content.size())

