from typing import Dict, List

from src.repository.elasticsearch_repository import ElasticSearchRepository
from src.repository.generate_embedding_repository import EmbeddingGeneratorRepository


class Movie:
    def __init__(
        self,
        elasticsearch_repo: ElasticSearchRepository,
        generate_embedding_repo: EmbeddingGeneratorRepository,
        movie_vector_index: str,
        movie_feature_index: str,
    ):
        self.elasticsearch_repo = elasticsearch_repo
        self.generate_embedding_repo = generate_embedding_repo
        self.movie_feature_index = movie_feature_index
        self.movie_vector_index = movie_vector_index

    def _update_vector_index(self, updated_items: Dict):
        movie_id = list(updated_items.keys())
        movie_values = list(updated_items.values())
        movie_feature = [
            f"{movie_value['genres']}|{movie_value['tags']}"
            for movie_value in movie_values
        ]
        # generate embedding
        generated_embeddings = self.generate_embedding_repo.get_generate(
            movie_features=movie_feature
        )
        # save it to elasticsearch
        docs = [
            {
                "movie_id": movie_id[idx],
                "movie_vector": generated_embeddings[movie_feature[idx]],
            }
            for idx in range(len(movie_id))
        ]
        self.elasticsearch_repo.insert_index(self.movie_vector_index, docs=docs)

    def _update_feature_index(self, updated_items: Dict):
        movie_id = list(updated_items.keys())
        movie_values = list(updated_items.values())

        docs = [
            {
                "movie_id": movie_id[idx],
                "title": movie_values[idx]["title"],
                "genres": movie_values[idx]["genres"],
                "tags": movie_values[idx]["tags"],
            }
            for idx in range(len(movie_id))
        ]
        self.elasticsearch_repo.insert_index(self.movie_feature_index, docs)

    def update(self, items):
        self._update_feature_index(items)
        self._update_vector_index(items)
