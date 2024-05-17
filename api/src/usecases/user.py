from typing import Dict, List

from src.repository.elasticsearch_repository import ElasticSearchRepository


class User:
    def __init__(
        self,
        elasticsearch_repo: ElasticSearchRepository,
        user_index: str,
    ):
        self.elasticsearch_repo = elasticsearch_repo
        self.user_index = user_index

    def _update_user_index(self, updated_items: Dict):
        user_id = list(updated_items.keys())
        user_values = list(updated_items.values())

        # save it to elasticsearch
        docs = [
            {
                "user_id": user_id[idx],
                "movie_id": user_values[idx]["movie_id"],
                "ratings": user_values[idx]["ratings"],
            }
            for idx in range(len(user_id))
        ]
        self.elasticsearch_repo.insert_index(
            self.user_index, docs=docs, use_as_id="user_id"
        )

    def update(self, items):
        self._update_user_index(items)
