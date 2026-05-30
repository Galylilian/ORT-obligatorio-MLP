from typing import List
import torch
import time

from transformers import pipeline

from src.settings import custom_logger


class ImageClassifier:
    """Class for handling image classification."""

    def __init__(self, model_id: str, batch_size: int):
        self.logger = custom_logger(self.__class__.__name__)
        self.model_id = model_id
        self.batch_size = batch_size

        self.device = 0 if torch.cuda.is_available() else -1
        self.logger.info(f"Using device: {self.device}")

        self.load_model()

    def load_model(self):
        """Load Hugging Face image classification pipeline."""
        self.logger.info(f"Loading image model {self.model_id}")

        self.pipeline = pipeline(
            task="image-classification",
            model=self.model_id,
            device=self.device
        )

        self.logger.info("Model loaded successfully!")

    def _create_batches(self, images: List[str]):
        for i in range(0, len(images), self.batch_size):
            yield images[i:i + self.batch_size]

    def predict(self, image_urls: List[str]):
        """Run inference on image URLs"""

        # ✅ Manejo de input vacío
        if not image_urls:
            return {
                "images": [],
                "model_id": self.model_id,
                "inference_time": 0.0
            }

        start = time.time()
        results = []

        for batch_urls in self._create_batches(image_urls):

            # ✅ HuggingFace maneja las URLs directamente
            predictions = self.pipeline(batch_urls)

            for url, pred in zip(batch_urls, predictions):
                top = pred[0]

                results.append({
                    "image_url": url,
                    "label": top["label"],
                    "score": float(top["score"]),
                    "metadata": {
                        "top_k": pred
                    }
                })

        end = time.time()

        self.logger.info(
            f"Processed {len(results)} images in {end - start:.3f}s"
        )

        return {
            "images": results,
            "model_id": self.model_id,
            "inference_time": round(end - start, 4)
        }