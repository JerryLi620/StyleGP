import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models.vgg import VGG16_Weights
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

def extract_feature(image, layer_indices=[0, 5, 10, 17]):
    # Define the image transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])

    # Load and transform the image
    image = transform(image).unsqueeze(0).to(device)
    # Extract feature maps
    feature_maps = []
    with torch.no_grad():
        for i, layer in enumerate(model.children()):
            image = layer(image)
            if i in layer_indices:
                feature_maps.append(image)
    return feature_maps


def gram_matrix(feature_maps):
    gram_matrices = []
    for feature_map in feature_maps:
        (b, c, h, w) = feature_map.size()  # b=batch size, c=number of feature maps, h=height, w=width
        features = feature_map.view(b * c, h * w)  # Reshape the feature map
        G = torch.mm(features, features.t())  # Compute the Gram matrix
        gram_matrices.append(G.div(b * c * h * w))  # Normalize the Gram matrix

    return gram_matrices


# style_image = Image.open("generated_style.jpg").convert("RGB")
# style_image2 = Image.open("dog.jpg").convert("RGB")
# feature_map = extract_feature(style_image)
# feature_map2 = extract_feature(style_image2)
# # print(feature_map[0].size())
# gram = gram_matrix(feature_map)
# gram2 = gram_matrix(feature_map2)
# style_loss = 0
# for i in range(len(gram)):
#     print(torch.mean(np.abs(gram2[i] - gram[i]))* 1e5)
#     style_loss += torch.mean(np.abs(gram2[i] - gram[i]))* 1e5
