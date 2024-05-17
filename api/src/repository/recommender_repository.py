import json
from typing import Dict, List, Tuple

import requests


class RecommenderRepository:
    def __init__(self, address):
        self.address = address

    def get_recommender(
        self,
        historical: Tuple[List[List[float]], List[float]],
        prediction_embeddings: List[List[float]],
        movie_id: List[str],
    ) -> List[str]:
        data = {
            "historical": historical,
            "prediction_embeddings": prediction_embeddings,
            "movie_id": movie_id,
        }
        headers = {"Content-Type": "application/json"}

        response = requests.post(self.address, headers=headers, data=json.dumps(data))
        return response
