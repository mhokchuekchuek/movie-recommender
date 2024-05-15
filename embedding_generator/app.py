import os
from typing import Dict, List

from fastapi import FastAPI
from src.usecases.engine import GeneratorEngine

app = FastAPI()
model_path = "src/model"
engine = GeneratorEngine(model_path)


@app.get("/")
async def root():
    return {"message": "Hello from Embedding Generator"}


@app.post("/generate")
async def process_strings(texts: List[str]) -> Dict[str, List[float]]:
    return engine.get_embedding_from_sentence(texts=texts)
