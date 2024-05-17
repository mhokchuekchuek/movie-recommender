from typing import Dict, List, Tuple

from src.repository.elasticsearch_repository import ElasticSearchRepository
from src.repository.recommender_repository import RecommenderRepository


class Recommender:
    def __init__(
        self,
        elasticsearch_repo: ElasticSearchRepository,
        recommender_repo: RecommenderRepository,
        vector_index: str,
        user_index: str,
        movie_index: str,
        threshold: int = 20,
    ):
        self.elasticsearch_repo = elasticsearch_repo
        self.recommender_repo = recommender_repo
        self.vector_index = vector_index
        self.user_index = user_index
        self.movie_index = movie_index
        self.threshold = threshold

    def _get_similar_movies(self, embeddings: List[List[float]]) -> List[str]:
        recommend = []
        for embedding in embeddings:
            sim_response = self.elasticsearch_repo.get_similar_movies(
                embeddings=embedding, index_name=self.vector_index
            )
            for res in sim_response:
                recommend.append(res["movie_id"])

        return list(set(recommend))

    def _get_sim_from_history_rating(
        self,
        historicals: Tuple[List[List[float]], List[float]],
        prediction_embeddings: List[List[float]],
        movie_id: List[str],
    ) -> List[str]:
        response = self.recommender_repo.get_recommender(
            historical=historicals,
            prediction_embeddings=prediction_embeddings,
            movie_id=movie_id,
        )
        return response.json()

    def get_histories(self, user_id: str) -> Dict[str, List[Dict]]:
        response = self.elasticsearch_repo.get_record_from_id(self.user_index, user_id)
        histories = response["movie_id"]
        return {"features": [{"histories": histories}]}

    def recommend(self, user_id: "str") -> List[str]:
        get_historical = self.elasticsearch_repo.get_record_from_id(
            index_name=self.user_index, id=user_id
        )
        movie_embeddings = self.elasticsearch_repo.get_all_record(
            index_name=self.vector_index
        )
        movie_features = self.elasticsearch_repo.get_all_record(
            index_name=self.movie_index
        )
        if len(get_historical["movie_id"]) >= self.threshold:
            prediction_embeddings = []
            historicals_embeddings = []
            ratings = []
            movie_id = []

            for idx, emb in enumerate(movie_embeddings):
                if emb["movie_id"] in get_historical["movie_id"]:
                    historicals_embeddings.append(emb["movie_vector"])
                    ratings.append(get_historical["ratings"][idx])
                else:
                    prediction_embeddings.append(emb["movie_vector"])
                    movie_id.append(emb["movie_id"])

            historicals = (historicals_embeddings, ratings)

            response = self._get_sim_from_history_rating(
                historicals=historicals,
                prediction_embeddings=prediction_embeddings,
                movie_id=movie_id,
            )

        else:
            embeddings = [
                emb["movie_vector"]
                for emb in movie_embeddings
                if emb["movie_id"] in get_historical["movie_id"]
            ]
            response = self._get_similar_movies(embeddings=embeddings)

        items_data = [{"id": movie_id} for movie_id in response]
        metadata = [
            movie_feature
            for movie_feature in movie_features
            if movie_feature["movie_id"] in response
        ]
        return items_data, metadata
