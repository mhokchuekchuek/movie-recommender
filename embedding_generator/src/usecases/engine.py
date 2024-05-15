from typing import Dict, List

import numpy as np
import torch
from src.service.generate import GeneratorService


class GeneratorEngine:
    def __init__(self, model_path: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = GeneratorService(model_path=model_path, device=device)

    def get_embedding_from_sentence(self, texts: List[str]) -> Dict[str, List[float]]:
        embeddings = self.generator.generate(texts)
        print("embedings", embeddings, flush=True)
        return {texts[index]: embeddings[index] for index in range(len(embeddings))}
