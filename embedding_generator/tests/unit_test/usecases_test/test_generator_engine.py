import unittest
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import torch
from src.usecases.engine import GeneratorEngine


class TestGeneratorService(unittest.TestCase):

    @patch("src.service.generate.GeneratorService")
    @patch("src.service.generate.GeneratorService.generate")
    @patch("src.service.generate.GeneratorService._load_model")
    def test_get_embedding_from_sentence(
        self, mock_load_model, mock_generate, mock_generator_service
    ):
        model_path = "test_model_path"
        texts = ["hello", "world"]
        mock_generate.return_value = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        expected_result = {"hello": [1.0, 2.0, 3.0], "world": [4.0, 5.0, 6.0]}

        engine = GeneratorEngine(model_path=model_path)
        result = engine.get_embedding_from_sentence(texts=texts)

        mock_generate.assert_called_once_with(texts)
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
