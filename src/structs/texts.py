from typing import Dict

from pydantic import BaseModel

from src.structs.labels import ClassificationLabel


class ScoresMetadata(BaseModel):
    """Metadata for the scores of the classification labels"""

    scores: Dict[ClassificationLabel, float]


class ClassifiedText(BaseModel):
    """Classified text with label and score"""

    text: str
    label: ClassificationLabel
    score: float
    metadata: ScoresMetadata
