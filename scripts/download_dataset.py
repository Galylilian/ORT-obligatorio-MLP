import os
import shutil
from roboflow import Roboflow
from dotenv import load_dotenv

# =============================
# CONFIG
# =============================
load_dotenv()

api_key = os.getenv("ROBOFLOW_API_KEY")

if not api_key:
    raise ValueError("⚠️ Falta ROBOFLOW_API_KEY")

# =============================
# DESCARGA
# =============================
rf = Roboflow(api_key=api_key)
project = rf.workspace("roboflow-rzlru").project("medical-c4r1c")
version = project.version(10)

print("⬇️ Descargando dataset...")

dataset = version.download("yolov8")

print("✅ Dataset descargado")

# =============================
# RUTA ORIGINAL (medical-c4r1c-10)
# =============================
dataset_path = dataset.location
print("📂 Dataset en:", dataset_path)

# =============================
# RUTA DESTINO (/data/raw)
# =============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
final_path = os.path.join(BASE_DIR, "data", "raw")

# =============================
# LIMPIAR DESTINO
# =============================
if os.path.exists(final_path):
    shutil.rmtree(final_path)

# =============================
# COPIAR TODO EL CONTENIDO
# =============================
shutil.copytree(dataset_path, final_path)

print("✅ Todo el dataset copiado a data/raw")