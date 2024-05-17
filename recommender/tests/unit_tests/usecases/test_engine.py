import unittest
from unittest.mock import patch

import numpy as np
from src.usecases.engine import RecommenderEngine


class TestRecommenderEngine(unittest.TestCase):

    @patch("sklearn.ensemble.RandomForestRegressor.fit")
    def test_rerank(self, mock_model_fit):
        mock_historical = (
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],  # X
            [1.0, 1.5, 2.0],  # y
        )
        recommender_engine = RecommenderEngine(mock_historical)
        mock_movie_id = ["mov_1", "mov_2"]
        mock_ratings = [1, 5]
        expected_rerank = ["mov_2", "mov_1"]
        expected_type = list
        rerank = recommender_engine._rerank(mock_movie_id, mock_ratings)

        self.assertEqual(rerank, expected_rerank)
        self.assertEqual(type(rerank), expected_type)
        self.assertTrue([True for i in rerank if type(i) == str])

    @patch("src.usecases.engine.RecommenderEngine._rerank")
    @patch("src.service.get_rating_score_from_user.RecommenderService.score")
    @patch("sklearn.ensemble.RandomForestRegressor.predict")
    @patch("sklearn.ensemble.RandomForestRegressor.fit")
    def test_recommend(
        self, mock_model_fit, mock_model_predict, mock_score, mock_rerank
    ):
        mock_prediction_embeddings = [[1], [1], [1]]
        mock_movie_id = ["mov_1", "mov_2", "mov_3"]
        mock_historical = (
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],  # X
            [1.0, 1.5, 2.0],  # y
        )
        recommender_engine = RecommenderEngine(mock_historical)

        recommender_engine.recommend(mock_prediction_embeddings, mock_movie_id)

        mock_score.assert_called_once_with(
            prediction_embeddings=mock_prediction_embeddings
        )
        mock_rerank.assert_called_once()


if __name__ == "__main__":
    unittest.main()
