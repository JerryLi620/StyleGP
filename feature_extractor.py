import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models.vgg import VGG16_Weights


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

def extract_feature(image):
    # Define the image transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

# image = Image.open("style_image.jpg").convert("RGB")
# feature_map = extract_feature(image)
# print(feature_map.size())
# gram = gram_matrix(feature_map)
# print(gram.size())
