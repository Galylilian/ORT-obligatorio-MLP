from fastapi import APIRouter, Request
from fastapi import Body

from src.settings import custom_logger
from src.structs.payload import TextsPayload, ResponseTextsPayload


logger = custom_logger("Classification Router")

classification_router = APIRouter()


@classification_router.post("/texts")
def classify_texts(
    request: Request, texts: TextsPayload = Body(...)
) -> ResponseTextsPayload:
    """
    Endpoint for classifying a list of texts

    Args:
        request: Request object
        texts: TextsPayload object containing the texts to classify

    Returns:
        ResponseTextsPayload object containing the classified texts
    """

    preprocessed_payload: TextsPayload = request.state.preprocessor.preprocess_texts(
        texts
    )
    response: ResponseTextsPayload = request.state.classifier.predict(
        preprocessed_payload
    )
    return response
