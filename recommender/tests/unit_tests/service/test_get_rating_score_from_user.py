import unittest
from unittest.mock import patch

import numpy as np
from src.service.get_rating_score_from_user import RecommenderService


class TestRecommenderService(unittest.TestCase):
    def mock_recommender_service(self):
        self.historical = (
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],  # X
            [1.0, 1.5, 2.0],  # y
        )
        return RecommenderService(historical=self.historical)

    @patch("sklearn.ensemble.RandomForestRegressor.fit")
    def test_get_model(self, mock_random_forest_fit):
        mock_recommender_service = self.mock_recommender_service()
        mock_recommender_service._get_model()

        mock_random_forest_fit.assert_called()

    @patch("sklearn.ensemble.RandomForestRegressor.predict")
    def test_inference(self, mock_predict):
        mock_prediction_embeddings = [[1]]
        mock_recommender_service = self.mock_recommender_service()

        mock_recommender_service.inference(mock_prediction_embeddings)
        mock_predict.assert_called_once_with(mock_prediction_embeddings)

    def test_postprocess(self):
        mock_predicted_rating = np.array([5, 5, 5])
        mock_recommender_service = self.mock_recommender_service()
        expected_result = [5, 5, 5]
        expected_type = list
        result = mock_recommender_service.postprocess(mock_predicted_rating)

        self.assertEqual(result, expected_result)
        self.assertEqual(type(result), expected_type)

    @patch("src.service.get_rating_score_from_user.RecommenderService.postprocess")
    @patch("src.service.get_rating_score_from_user.RecommenderService.inference")
    def test_score(self, mock_inference, mock_postprocess):
        mock_prediction_embeddings = [[1], [1], [1]]
        mock_recommender_service = self.mock_recommender_service()

        mock_recommender_service.score(mock_prediction_embeddings)

        mock_inference.assert_called_once_with(
            prediction_embeddings=mock_prediction_embeddings
        )
        mock_postprocess.assert_called_once()


if __name__ == "__main__":
    unittest.main()
