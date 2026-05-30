from fastapi import APIRouter, Request, Body
from typing import List

classification_router = APIRouter()


@classification_router.post("/images")
def classify_images(
    request: Request,
    images: List[str] = Body(...)
):
    """
    Endpoint for classifying a list of images

    Args:
        request: Request object
        images: List of image URLs

    Returns:
        Dict with classification results
    """

    # NO hay preprocesador (por ahora)
    response = request.state.image_classifier.predict(images)

    return response