from typing import Dict

from elasticsearch import Elasticsearch


class CreateElasticSearchIndex:
    def __init__(
        self,
        elasticsearch_address: str,
        vector_index_name: str,
        movie_index_name: str,
        user_index_name: str,
    ):
        self.client = Elasticsearch(elasticsearch_address)

        self.vector_index_name = vector_index_name
        self.movie_index_name = movie_index_name
        self.user_index_name = user_index_name
        self.mappings = self._create_mappings()

        self.create_index(index_name=self.vector_index_name)
        self.create_index(index_name=self.movie_index_name)
        self.create_index(index_name=self.user_index_name)

    def _create_mappings(self) -> Dict:
        vector_index_mappings = {
            "properties": {
                "movie_id": {"type": "keyword"},
                "movie_vector": {
                    "type": "dense_vector",
                    "dims": 768,
                    "index": "true",
                    "similarity": "cosine",
                },
            }
        }

        movie_index_mappings = {
            "properties": {
                "movie_id": {"type": "keyword"},
                "title": {"type": "text"},
                "genres": {"type": "keyword"},
                "tags": {"type": "keyword"},
            }
        }

        user_index_mappings = {
            "properties": {
                "movie_id": {"type": "keyword"},
                "ratings": {"type": "float"},
            }
        }
        return {
            self.vector_index_name: vector_index_mappings,
            self.movie_index_name: movie_index_mappings,
            self.user_index_name: user_index_mappings,
        }

    def create_index(self, index_name: str):
        if not self._is_existed_index(index_name=index_name):
            self.client.indices.create(
                index=index_name, mappings=self.mappings[index_name]
            )

    def _is_existed_index(self, index_name: str) -> bool:
        if self.client.indices.exists(index=index_name):
            return True
        return False

    def delete_index(self, index_name: str) -> bool:
        if self.is_existed_index(index_name):
            self.client.indices.delete(index=index_name)
