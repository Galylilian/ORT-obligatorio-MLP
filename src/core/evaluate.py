import torch
from sklearn.metrics import classification_report
from src.settings.config import MODEL_PATH, DEVICE
from src.core.model import get_model
from src.utils.metrics import compute_metrics
def load_model():
    model = get_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def evaluate(model, loader):
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)

            outputs = model(images)
            preds = outputs.argmax(1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    print(classification_report(y_true, y_pred))
    
metrics = compute_metrics(y_true, y_pred)
print(metrics)
