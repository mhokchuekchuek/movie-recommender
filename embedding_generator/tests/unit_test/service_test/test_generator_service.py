import unittest
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import torch
from src.service.generate import GeneratorService


class TestGeneratorService(unittest.TestCase):

    @patch("src.service.generate.SentenceTransformer")
    def test_load_model(self, mock_sentence_transformer):
        model_path = "test_model_path"
        device = "cpu"
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model

        service = GeneratorService(model_path=model_path, device=device)

        mock_sentence_transformer.assert_called_once_with(model_name_or_path=model_path)
        self.assertEqual(service.model, mock_model)
        self.assertEqual(service.device, device)

    @patch("src.service.generate.SentenceTransformer")
    def test_inference(self, mock_sentence_transformer):
        model_path = "test_model_path"
        device = "cpu"
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model
        service = GeneratorService(model_path=model_path, device=device)

        texts = ["hello", "world"]
        expected_output = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mock_model.encode.return_value = expected_output

        result = service._inference(texts)

        mock_model.encode.assert_called_once_with(texts)
        np.testing.assert_array_equal(result, expected_output)

    @patch("src.service.generate.GeneratorService._load_model")
    def test_postprocess(self, mock_load_model):
        service = GeneratorService(model_path="path_to_model", device="cpu")
        inferenced_input = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        expected_output = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]

        result = service._postprocess(inferenced_input)

        self.assertEqual(result, expected_output)

    @patch("src.service.generate.SentenceTransformer")
    def test_generate(self, mock_sentence_transformer):
        model_path = "test_model_path"
        device = "cpu"
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model
        service = GeneratorService(model_path=model_path, device=device)

        texts = ["hello", "world"]
        inferenced_output = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        expected_output = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        mock_model.encode.return_value = inferenced_output

        result = service.generate(texts)

        mock_model.encode.assert_called_once_with(texts)
        self.assertEqual(result, expected_output)


if __name__ == "__main__":
    unittest.main()
