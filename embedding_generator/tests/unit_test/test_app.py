import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient


class TestGeneratorEngine(unittest.TestCase):
    @patch("src.usecases.engine.GeneratorEngine.get_embedding_from_sentence")
    @patch("src.service.generate.GeneratorService")
    @patch("src.service.generate.GeneratorService.generate")
    @patch("src.service.generate.GeneratorService._load_model")
    def create_client(
        self, mock_load_model, mock_generate, mock_generator_service, mock_get_embedding
    ):
        from app import app

        client = TestClient(app)
        return client

    def test_root_endpoint(self):
        client = self.create_client()
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Hello from Embedding Generator"})

    @patch("src.usecases.engine.GeneratorEngine.get_embedding_from_sentence")
    def test_generate_endpoint(self, mock_get_embedding):
        client = self.create_client()
        mock_texts = ["test sentence"]
        mock_embeddings = [0.1, 0.2, 0.3]
        mock_get_embedding.return_value = {"test sentence": mock_embeddings}
        expected_result = {"test sentence": mock_embeddings}
        expected_status_code = 200

        response = client.post("/generate", json=["test sentence"])
        self.assertEqual(response.status_code, expected_status_code)
        self.assertEqual(response.json(), expected_result)
        mock_get_embedding.assert_called_once_with(texts=mock_texts)


if __name__ == "__main__":
    unittest.main()
