import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models.vgg import VGG16_Weights

# Load the pretrained VGG16 model
model = models.vgg16(weights=VGG16_Weights.DEFAULT)

# We will use the model as a feature extractor, so we don't need the final classification layers.
# We only take the feature layer, average pooling layer, and one fully-connected layer.
model.classifier = torch.nn.Sequential(*[model.classifier[i] for i in range(4)])

# Ensure the model is in evaluation mode
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and transform the image
image = Image.open("style_image.jpg").convert('RGB') 
image = transform(image).unsqueeze(0)  # Add batch dimension

# Check if GPU is available and if so, use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image = image.to(device)
model = model.to(device)

# Extract features
with torch.no_grad():
    features = model(image)

# The 'features' tensor contains the feature vector for the image
print(features)

