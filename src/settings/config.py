
import torch
import os

# Ruta del modelo desde variable de entorno
MODEL_PATH = os.getenv("MODEL_PATH", "models/resnet18.pth")

# Dispositivo (CPU o GPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
