import json

import requests


def test_root_endpoint():
    response = requests.get("http://recommender:8001/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello from Recommender"}


def test_recommend_endpoint():
    historical = (
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],  # Historical feature data
        [0.5, 0.7],  # Historical target data
    )
    prediction_embeddings = [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]
    movie_id = ["movie1", "movie2"]

    data = {
        "historical": historical,
        "prediction_embeddings": prediction_embeddings,
        "movie_id": movie_id,
    }

    headers = {"Content-Type": "application/json"}
    response = requests.post(
        "http://recommender:8001/recommend", headers=headers, data=json.dumps(data)
    )

    result = response.json()

    assert response.status_code == 200
    assert isinstance(result, list)
    assert all(isinstance(x, str) for x in result)
