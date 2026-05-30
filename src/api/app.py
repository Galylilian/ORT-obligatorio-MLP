from fastapi import FastAPI

from src.api.routers.predict import router as predict_router
from src.api.routers.predict_yolo import router as predict_yolo_router
from src.api.routers.gradcam import router as gradcam_router
from src.api.routers.health import health_router

app = FastAPI()

# incluir routers
app.include_router(health_router, tags=["Health"])
app.include_router(predict_router, tags=["CNN"])
app.include_router(predict_yolo_router, tags=["YOLO"])
app.include_router(gradcam_router, tags=["Explainability"])