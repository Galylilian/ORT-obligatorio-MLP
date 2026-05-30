from typing import List

from pydantic import BaseModel

from src.structs.texts import ClassifiedText


class TextsPayload(BaseModel):
    """Input payload for the classification endpoint"""

    texts: List[str]


class ResponseTextsPayload(BaseModel):
    """Response payload for the classification endpoint"""

    texts: List[ClassifiedText]
    model_id: str
