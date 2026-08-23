"""
Inference Snippet for Part 2 & Part 3
Function to load the trained model and perform single-image prediction.
"""

import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

# 10 Fashion-MNIST Classes (Flipkart Catalog apparel / footwear / accessories)
CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

class ClassifierHead(nn.Module):
    def __init__(self, in_features=512, num_classes=10, dropout=0.2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
    def forward(self, x):
        return self.fc(x)

class CompleteProductClassifier(nn.Module):
    def __init__(self, backbone_model, classifier_head):
        super().__init__()
        self.backbone = backbone_model
        self.head = classifier_head
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

_model_cache = None
_transform_cache = None

def get_inference_pipeline(weights_path: str = "models/product_classifier.pt"):
    """
    Loads and caches the model weights and preprocessing pipeline.
    """
    global _model_cache, _transform_cache
    if _model_cache is not None:
        return _model_cache, _transform_cache
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Backbone
    weights = ResNet18_Weights.DEFAULT
    backbone = resnet18(weights=weights)
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.fc = nn.Identity()
    
    # 2. Head
    head = ClassifierHead(in_features=512, num_classes=10)
    
    # 3. Assemble
    model = CompleteProductClassifier(backbone, head)
    
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"Model weights not found at {weights_path}")
        
    model.to(device)
    model.eval()
    
    # Preprocessing (matches training resolution)
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    _model_cache = (model, device)
    _transform_cache = transform
    return _model_cache, _transform_cache

def classify_product_image(image_path: str, weights_path: str = "models/product_classifier.pt") -> dict:
    """
    Single-image prediction function for Part 3 tool.
    
    Args:
        image_path (str): Path to input image file (.png, .jpg, etc.)
        weights_path (str): Path to trained product_classifier.pt weights
        
    Returns:
        dict: {
            'predicted_class': str,
            'confidence': float,
            'class_probabilities': dict[str, float]
        }
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    (model, device), transform = get_inference_pipeline(weights_path)
    
    # Load image
    img = Image.open(image_path)
    tensor_img = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(tensor_img)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        
    predicted_idx = int(probs.argmax())
    predicted_class = CLASSES[predicted_idx]
    confidence = float(probs[predicted_idx])
    
    class_probabilities = {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}
    
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "class_probabilities": class_probabilities
    }

if __name__ == "__main__":
    import sys
    test_img = "data/sample_images/00_ankle_boot.png" if len(sys.argv) < 2 else sys.argv[1]
    if os.path.exists(test_img):
        res = classify_product_image(test_img)
        print(f"Prediction for {test_img}:")
        print(f"  Class: {res['predicted_class']} (Confidence: {res['confidence']:.4f})")
    else:
        print(f"Sample test image {test_img} not found yet.")
