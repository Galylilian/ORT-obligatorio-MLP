from src.settings import custom_logger
from src.structs.payload import TextsPayload


class Preprocessor:
    """Class for handling the preprocessing operations of the text data"""

    def __init__(self) -> None:
        self.logger = custom_logger(self.__class__.__name__)

    def preprocess_texts(self, payload: TextsPayload) -> TextsPayload:
        """
        Method for preprocessing a batch of text data

        Args:
            payload: TextsPayload object containing the texts to preprocess

        Returns:
            TextsPayload object containing the preprocessed texts
        """
        return TextsPayload(
            texts=[self.preprocess_text(text) for text in payload.texts]
        )

    def preprocess_text(self, text: str) -> str:
        """
        Method for preprocessing a single text

        Args:
            text: Text to preprocess

        Returns:
            Preprocessed text
        """
        new_text = []
        for t in text.split(" "):       #Reemplaza menciones y links por tokens genéricos.
            t = "@user" if t.startswith("@") and len(t) > 1 else t # Es el mismo preprocesamiento que se hizo para entrenar cardiffnlp/twitter-roberta-base-sentiment-latest
            t = "http" if t.startswith("http") else t
            new_text.append(t)
        return " ".join(new_text)
