# Embedding generator
This is REST API that use `bert-base-nli-mean-tokens` for create embedding

# API Endpoints
|endpoint|input|output|
|--------|-----|------|
|`generate`|text that do embedding <br>`type:`List[str]|return as Dict[str, List[float]] <br> which key is text and value is embedding (list of float)

### example
request
```python
import requests

# Define the URL of the FastAPI endpoint
url = "http://127.0.0.1:8000/generate"

# Define the payload (list of strings)
payload = ["apple", "banana", "cherry"]

# Make the POST request
response = requests.post(url, json=payload)

# Print the response (JSON)
print(response.json())
```
response
```
{
    "apple": [0.133,...],
    "banana": [0.11, ...],
    "cherry":[0.13, ...]
}
```

# Run API Locally
### Prerequisite
1. Docker

## Setup
1. Download Model
```
make init_model
```
2.Compose up the containers
```
docker compose up --force-recreate --build
```
## Tests
### Integration Tetst & Unit Test
```
 make test
```