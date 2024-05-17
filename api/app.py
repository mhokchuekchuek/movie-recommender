import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from pydantic import BaseModel
from src.adapter.elasticsearch import ElasticsearchAdapter
from src.repository.elasticsearch_repository import ElasticSearchRepository
from src.repository.generate_embedding_repository import EmbeddingGeneratorRepository
from src.repository.recommender_repository import RecommenderRepository
from src.usecases.movies import Movie
from src.usecases.recommend import Recommender
from src.usecases.user import User

load_dotenv()

ELASTICSEARCH_CLIENT = os.environ.get("ELASTICSEARCH_CLIENT", "http://localhost:9200")
EMBEDDING_GENERATOR_ENDPOINT = os.environ.get(
    "EMBEDDING_GENERATOR_ENDPOINT", "http://localhost:8000/generate"
)
RECOMMENDER_ENDPOINT = os.environ.get(
    "RECOMMENDER_ENDPOINT", "http://localhost:8001/recommend"
)
VECTOR_INDEX = os.environ.get("VECTOR_INDEX", "movie_vector_1")
MOVIE_INDEX = os.environ.get("MOVIE_INDEX", "movie_index_1")
USER_INDEX = os.environ.get("USER_INDEX", "user_index_1")

# adapter
elasticsearch_adapter = ElasticsearchAdapter(ELASTICSEARCH_CLIENT)

# repo
elasticsearch_repo = ElasticSearchRepository(elasticsearch_adapter)
embedding_repo = EmbeddingGeneratorRepository(EMBEDDING_GENERATOR_ENDPOINT)
recom_repo = RecommenderRepository(RECOMMENDER_ENDPOINT)

# usecase
update_embedding = Movie(elasticsearch_repo, embedding_repo, VECTOR_INDEX, MOVIE_INDEX)
update_user = User(elasticsearch_repo, USER_INDEX)
reommender = Recommender(
    elasticsearch_repo=elasticsearch_repo,
    recommender_repo=recom_repo,
    user_index=USER_INDEX,
    vector_index=VECTOR_INDEX,
    movie_index=MOVIE_INDEX,
)

app = FastAPI()


class Rating(BaseModel):
    movie_id: Any
    ratings: List[Any]


class Movie(BaseModel):
    title: str
    genres: str
    tags: str


@app.get("/recommendations")
def get_movie_recommend(
    user_id: str, returnMetadata: Optional[bool] = Query(False, alias="returnMetadata")
):
    items_data, metadata = reommender.recommend(user_id=user_id)
    if returnMetadata:
        return {"items": metadata}
    else:
        return {"items": items_data}


@app.get("/features")
def get_history(user_id: str):
    return reommender.get_histories(user_id=user_id)


@app.post("/update_ratings")
async def update_ratings(ratings_input: Dict[str, Rating]):
    keys = ratings_input.keys()
    values = [
        {"movie_id": rat.movie_id, "ratings": rat.ratings}
        for rat in ratings_input.values()
    ]
    ratings_input = dict(zip(keys, values))
    update_user.update(ratings_input)


@app.post("/update_movie")
async def update_movie(movie_input: Dict[str, Movie]):
    keys = movie_input.keys()
    values = [
        {"title": mov.title, "genres": mov.genres, "tags": mov.tags}
        for mov in movie_input.values()
    ]
    movie_input = dict(zip(keys, values))
    update_embedding.update(movie_input)
