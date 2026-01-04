import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

from torchvision.models import resnet18, ResNet18_Weights

def get_coin_model(num_classes=6):
    """
    Vrátí ResNet18 model předtrénovaný na ImageNet,
    s upravenou finální vrstvou pro klasifikaci mincí.
    """
    # Načtení předtrénovaného ResNet18
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    
    # Zmrazení dřívějších vrstev (volitelné, ale dobré pro malé datasety)
    # for param in model.parameters():
    #     param.requires_grad = False
        
    # Nahrazení finální plně propojené (Fully Connected) vrstvy
    # ResNet18 má 512 vstupních features do fc
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def load_model(model_path=None, num_classes=6):
    """
    Načte architekturu modelu a váhy (pokud jsou zadány).
    """
    model = get_coin_model(num_classes=num_classes)
    
    if model_path:
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print(f"Model loaded from {model_path}")
        except FileNotFoundError:
            print(f"Model file {model_path} not found, using initialized weights.")
    else:
        print("No model path provided, using pre-trained ImageNet weights.")
        
    model.eval()
    return model
