import os
import torch

from src.core.model import get_model
from src.data.dataset import get_dataloaders

# =============================
# DATOS
# =============================
train_loader, test_loader = get_dataloaders()

# =============================
# DISPOSITIVO
# =============================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Usando dispositivo: {device}")

# =============================
# MODELO
# =============================
model = get_model().to(device)

# =============================
# ENTRENAMIENTO
# =============================
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

for epoch in range(2):   # solo 2 epochs para demo
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: {total_loss:.4f}")

# =============================
# GUARDAR MODELO ✅
# =============================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

model_file = os.path.join(models_dir, "resnet18.pth")

import pathlib

models_dir = pathlib.Path(BASE_DIR) / "models"
models_dir.mkdir(parents=True, exist_ok=True)

model_file = models_dir / "resnet18.pth"


# guardar
torch.save(model.state_dict(), str(model_file))

print(f"✅ Modelo guardado en: {model_file}")
