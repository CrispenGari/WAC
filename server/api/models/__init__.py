import torch
import os
from torchvision import transforms
from torch.nn import functional as F
import numpy as np
from api.types import *

# torch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# allowed images 
allowed_extensions = ['.jpg', '.JPG', '.png' ,'.PNG' ,'.jpeg' ,'.JPEG']

# Model names
MODEL_NAME = "wild-animal-classification.pt"

# Model paths
PYTORCH_WAC_MODEL_PATH = os.path.join(os.getcwd(),
                                      f"api/models/pytorch/static/{MODEL_NAME}"
                                      )
    
# Classes
classes = ['cheetah', 'fox', 'hyena', 'lion', 'tiger', 'wolf']
means = [0.5059, 0.4904, 0.4246]
stds= [0.2292, 0.2269, 0.2292]

pretrained_size = 224
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds= [0.229, 0.224, 0.225]

test_transforms = transforms.Compose([
  transforms.Resize(pretrained_size),
  transforms.CenterCrop(pretrained_size),
  transforms.ToTensor(),
  transforms.Normalize(mean = pretrained_means, 
                    std = pretrained_stds)
])

def preprocess_img(img):
    """
    takes in a pillow image and pre process it
    """
    img = test_transforms(img)
    return img

def predict(model, image, device):
    image = image.unsqueeze(dim=0).to(device)
    preds, _ = model(image)
    preds = F.softmax(preds, dim=1).detach().cpu().numpy().squeeze()
    predicted_label = np.argmax(preds)
    predictions = [
        Prediction(
            label = i,
            class_name = classes[i],
            probability = np.round(preds[i], 2)
        ) for i, _ in enumerate(preds)
    ]
    predicted = Prediction(
        label = predicted_label,
        class_name = classes[predicted_label],
        probability = np.round(preds[predicted_label], 2)
    )
    return Response(
        top_prediction = predicted,
        predictions = predictions
    )