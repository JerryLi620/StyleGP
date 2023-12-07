import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models.vgg import vgg16_bn
import numpy as np

# Load the pretrained VGG16 model with batch normalization
model = models.vgg16_bn(pretrained=True)

# Modify the model to return feature maps from a specific convolutional layer
# For example, the output of the fifth convolutional block
model.features = torch.nn.Sequential(*list(model.features.children())[:])

# Freeze all the layers of the model
for param in model.parameters():
    param.requires_grad = False

# Set the model to evaluation mode
model.eval()

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def extract_feature(image):
    # Define the image transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])

    # Load and transform the image
    image = transform(image).unsqueeze(0).to(device)

    # Extract feature map
    with torch.no_grad():
        feature_maps = model(image)

    return feature_maps


def gram_matrix(feature_map):
    (b, c, h, w) = feature_map.size()  # b=batch size, c=number of feature maps, h=height, w=width
    features = feature_map.view(b * c, h * w)  # Reshape the feature map
    G = torch.mm(features, features.t())  # Compute the Gram matrix
    # Normalize the values of the Gram matrix
    return G.div(b * c * h * w)

# style_image = Image.open("generated_style.jpg").convert("RGB")
# style_image2 = Image.open("dog.jpg").convert("RGB")
# feature_map = extract_feature(style_image)
# feature_map2 = extract_feature(style_image2)
# print(feature_map.size())
# gram = gram_matrix(feature_map)
# gram2 = gram_matrix(feature_map2)
# print(gram.size())
# print(torch.mean(np.abs(gram - gram2))*1e6)
