from typing import Dict, List, Union

import numpy as np
from sentence_transformers import SentenceTransformer


class GeneratorService:
    def __init__(self, model_path: str, device: str):
        self.device = device
        self.model = self._load_model(model_path=model_path)

    def _load_model(self, model_path: str):
        return SentenceTransformer(model_name_or_path=model_path)

    def _inference(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts)

    def _postprocess(self, inferenced: np.ndarray) -> List[List[float]]:
        return inferenced.tolist()

    def generate(self, texts: List[str]) -> List[List[float]]:
        inferenced_input = self._inference(texts=texts)
        return self._postprocess(inferenced=inferenced_input)
