from fastapi import APIRouter, UploadFile
import os

from src.core.yolo_model import predict_yolo
from src.utils.logger import get_logger
from src.utils.files import save_uploaded_file, delete_file

router = APIRouter()

logger = get_logger("predict_yolo")


@router.post("/predict_yolo")
async def predict(file: UploadFile):
    logger.info(f"Request YOLO recibido: {file.filename}")

    try:
        # guardar archivo temporal
        path = save_uploaded_file(file)

        logger.info(f"Imagen guardada en: {path}")

        # predicción
        pred = predict_yolo(path)

        logger.info(f"Predicción YOLO realizada: {pred}")

        # limpiar archivo
        delete_file(path)

        return {
            "prediction": pred,
            "label": "acostado" if pred == 1 else "no_acostado"
        }

    except Exception as e:
        logger.error(f"Error en YOLO: {str(e)}")

        return {
            "error": "Error en predicción YOLO"
        }
