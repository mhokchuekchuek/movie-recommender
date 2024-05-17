import os
from typing import Dict, List, Tuple

from fastapi import FastAPI
from src.usecases.engine import RecommenderEngine

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello from Recommender"}


@app.post("/recommend")
async def recommend(
    historical: Tuple[List[List[float]], List[float]],
    prediction_embeddings: List[List[float]],
    movie_id: List[str],
) -> List[str]:
    recommender_engine = RecommenderEngine(historical=historical)
    return recommender_engine.recommend(
        prediction_embeddings=prediction_embeddings, movie_id=movie_id
    )
