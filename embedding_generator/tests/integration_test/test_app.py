import unittest

import requests


def test_root_endpoint():
    response = requests.get("http://embedding-generator:8000/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello from Embedding Generator"}


def test_generate_endpoint():
    response = requests.post(
        "http://embedding-generator:8000/generate", json=["test sentence"]
    )
    result = response.json()

    assert response.status_code == 200
    assert isinstance(result, dict)
    assert "test sentence" in result
    assert isinstance(result["test sentence"], list)
    assert all(isinstance(x, float) for x in result["test sentence"])
