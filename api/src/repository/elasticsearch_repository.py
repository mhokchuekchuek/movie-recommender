from typing import Dict, List

from src.adapter.elasticsearch import ElasticsearchAdapter


class ElasticSearchRepository:
    def __init__(self, client: ElasticsearchAdapter, size: int = 10000):
        self.client = client
        self.insert = self.client.insert()
        self.search = self.client.search()
        self.get = self.client.get()
        self.update = self.client.update()
        self.size = size

    def get_all_record(self, index_name: str) -> List[Dict]:
        search_response = self.search(
            index=index_name, body={"query": {"match_all": {}}}, size=self.size
        )
        return [record["_source"] for record in search_response["hits"]["hits"]]

    def get_record_from_id(self, index_name: str, id: str) -> Dict:
        search_response = self.get(index=index_name, id=id)
        return search_response["_source"]

    def update_recommend(self, index_name: str, id: str, recommend_items: List[str]):
        self.update(
            index=index_name, id=id, body={"doc": {"recommend": recommend_items}}
        )

    def get_similar_movies(self, index_name: str, embeddings: List[float]) -> Dict:
        search_response = self.search(
            index=index_name,
            knn={
                "field": "movie_vector",
                "query_vector": embeddings,
                "k": 10,
                "num_candidates": 10,
            },
        )
        return [record["_source"] for record in search_response["hits"]["hits"]]

    def insert_index(
        self, index_name: str, docs: List[Dict], use_as_id: str = "movie_id"
    ):
        for doc in docs:
            self.insert(index=index_name, id=doc[use_as_id], body=doc)
