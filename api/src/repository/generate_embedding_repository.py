from typing import Dict, List

import requests


class EmbeddingGeneratorRepository:
    def __init__(self, address):
        self.address = address

    def get_generate(self, movie_features: List[str]) -> Dict[str, List[float]]:
        response = requests.post(self.address, json=movie_features)
        return response.json()
