import torch
from PIL import Image
from torchvision import transforms

from src.core.model import get_model


def test_inference_runs():
    """
    Verifica que el modelo puede hacer inferencia sin fallar.
    """

    model = get_model()
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # imagen dummy negra
    img = Image.new("RGB", (224, 224))

    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)

    pred = output.argmax().item()

    assert pred in [0, 1]