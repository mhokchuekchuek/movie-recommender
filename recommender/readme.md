# Embedding generator
This is REST API that use embedding from [embedding_generator](https://github.com/mhokchuekchuek/movie-recommender/tree/main/embedding_generator) to predict rating from historical.

# API Endpoints
|endpoint|input|output|
|--------|-----|------|
|`/recommend`|`historical:` the tuple that contains historical movie embedding and historical ratings.<br>`prediction_embeddings:`the movie embedding for predict rating<br>`movie_id:`the movie_id for predict rating|return as list of recommened movie that sort from high to low ratings

### example
request
```python
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
```
response
```
["movie2", "movie1"]
```

# Run API Locally
### Prerequisite
1. Docker

## Setup
0. Install dependencies
```
pip install -r requirements.txt
```
1. Download Model
```
make init_model
```
2. Compose up the containers
```
docker compose up --force-recreate --build
```
## Tests
### Integration Tetst & Unit Test
```
 make test
```