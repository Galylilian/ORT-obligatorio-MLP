from enum import Enum


class ClassificationLabel(str, Enum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"

    def __str__(self):
        return str(self.value)

    @classmethod
    def from_str(cls, label: str):
        if label == "negative":
            return cls.NEGATIVE
        elif label == "neutral":
            return cls.NEUTRAL
        elif label == "positive":
            return cls.POSITIVE
        else:
            raise ValueError(f"Invalid label: {label}")
