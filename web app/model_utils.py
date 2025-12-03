import torch
import torch.nn as nn
import timm
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# --- 1. The Model Architecture (Must match your notebook exactly) ---
class XceptionClassifier(nn.Module):
    def __init__(self, num_classes: int = 1, pretrained: bool = False):
        super().__init__()
        self.xception = timm.create_model(
            'xception',
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
        )
        in_features = self.xception.num_features
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )
        self.gradients = None
        self.activations = None
    
    def forward(self, x):
        x = self.xception(x)
        x = self.classifier(x)
        return x
    
    def forward_with_cam(self, x):
        x = self.xception(x)
        self.activations = x
        if x.requires_grad:
            x.register_hook(self.save_gradient)
        x = self.classifier(x)
        return x
    
    def save_gradient(self, grad):
        self.gradients = grad

# --- 2. Grad-CAM Logic ---
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def generate_cam(self, input_tensor, target_class=None):
        # Forward pass
        score = self.model.forward_with_cam(input_tensor)
        
        if target_class is None:
            target_class = torch.sigmoid(score).round().long().item()
        
        output = score if int(target_class) == 1 else -score
        
        self.model.zero_grad()
        output.backward(retain_graph=True)
        
        gradients = self.model.gradients
        activations = self.model.activations
        
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1).squeeze()
        cam = torch.relu(cam)
        
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.detach().cpu().numpy(), torch.sigmoid(score).item()

def get_overlay(original_image, activation_map):
    """Merges the Heatmap with the Original Image"""
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)
    
    # Resize heatmap to image size
    heatmap = cv2.resize(activation_map, (original_image.shape[1], original_image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Blend
    overlayed = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)
    return heatmap, overlayed

# --- 3. Transforms (Validation only) ---
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])