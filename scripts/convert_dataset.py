import os
import shutil

# ruta REAL dentro del proyecto
base_path = "data/raw"
output_path = "data/processed"

splits = ["train", "valid", "test"]

for split in splits:
    img_dir = os.path.join(base_path, split, "images")
    label_dir = os.path.join(base_path, split, "labels")

    pos_dir = os.path.join(output_path, split, "acostado")
    neg_dir = os.path.join(output_path, split, "no_acostado")

    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(neg_dir, exist_ok=True)

    for file in os.listdir(img_dir):
        img_path = os.path.join(img_dir, file)
        label_path = os.path.join(label_dir, file.replace(".jpg", ".txt"))

        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            shutil.copy(img_path, os.path.join(pos_dir, file))
        else:
            shutil.copy(img_path, os.path.join(neg_dir, file))

print("✅ Dataset convertido correctamente")
