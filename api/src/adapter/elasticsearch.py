from typing import Dict, List

from elasticsearch import Elasticsearch


class ElasticsearchAdapter:
    def __init__(self, elasticsearch_address: str):
        self.client = Elasticsearch(elasticsearch_address)

    def _is_existed_index(self, index_name: str) -> bool:
        if self.client.indices.exists(index=index_name):
            return True
        return False

    def delete_index(self, index_name: str) -> bool:
        if self.is_existed_index(index_name):
            self.client.indices.delete(index=index_name)

    def insert(self):
        return self.client.index

    def search(self):
        return self.client.search

    def ping(self) -> bool:
        return self.client.ping()

    def update(self) -> bool:
        return self.client.update

    def get(self):
        return self.client.get

