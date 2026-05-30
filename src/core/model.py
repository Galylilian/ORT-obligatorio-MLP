import torch.nn as nn
import torchvision.models as models

def get_model():
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 2)

    return model