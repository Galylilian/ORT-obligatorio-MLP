from contextlib import asynccontextmanager
import os
import sys
from typing import Dict, Any, AsyncGenerator

sys.path.append(os.getcwd())

import uvicorn

from src.api.routers import init_routers
from src.core.classification import ImageClassifier
from src.settings import custom_logger, SettingsManager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


logger = custom_logger("API")


# Context manager: inicializa dependencias al startup
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Function for loading image classifier and settings on startup
    """
    try:
        logger.info("Starting up application...")

        # Config
        settings = SettingsManager()

        # SOLO modelo de imágenes
        image_classifier = ImageClassifier(
            model_id=settings.IMAGE_MODEL_ID,
            batch_size=settings.BATCH_SIZE,
        )

        logger.info("Application startup complete")

        # 👇 disponible en request.state
        yield {
            "settings": settings,
            "image_classifier": image_classifier,
        }

    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

    finally:
        logger.info("Shutting down application...")


# ✅ Inicialización de FastAPI
app = FastAPI(
    title="practico-4-2026 (Image Classification)",
    description="API de clasificación de imágenes usando Transformers",
    version="0.1.0",
    lifespan=lifespan,
)


# ✅ Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ✅ Routers (acá debe estar /classification/images)
init_routers(app)


# ✅ Entrypoint
if __name__ == "__main__":
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
    )
