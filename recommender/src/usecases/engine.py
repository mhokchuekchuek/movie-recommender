from typing import Dict, List, Tuple

from src.service.get_rating_score_from_user import RecommenderService


class RecommenderEngine:
    def __init__(
        self,
        historical: Tuple[List[List[float]], List[float]],
    ):
        self.recommender_service = RecommenderService(historical=historical)

    def _rerank(self, movie_id: List[str], ratings: List[float]) -> List[str]:
        _sort_movie_id_by_rating = list(zip(ratings, movie_id))
        desc_movie_id_by_rating = sorted(
            _sort_movie_id_by_rating, key=lambda x: x[0], reverse=True
        )
        return [item[1] for item in desc_movie_id_by_rating]

    def recommend(self, prediction_embeddings: List[List[float]], movie_id: List[str]):
        ratings = self.recommender_service.score(
            prediction_embeddings=prediction_embeddings
        )
        return self._rerank(movie_id=movie_id, ratings=ratings)
