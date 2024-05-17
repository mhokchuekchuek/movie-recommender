from typing import List, Tuple

from sklearn.ensemble import RandomForestRegressor


class RecommenderService:
    def __init__(self, historical: Tuple[List[List[float]], List[float]]):
        self.historical = historical
        self.model = self._get_model()

    def _get_model(self):
        X, y = self.historical
        regressor = RandomForestRegressor(max_depth=128, random_state=0)
        regressor.fit(X, y)
        return regressor

    def inference(self, prediction_embeddings: List[List[float]]):
        return self.model.predict(prediction_embeddings)

    def postprocess(self, predicted_ratings) -> List[float]:
        return predicted_ratings.tolist()

    def score(self, prediction_embeddings: List[List[float]]) -> List[float]:
        predicted_ratings = self.inference(prediction_embeddings=prediction_embeddings)
        return self.postprocess(predicted_ratings=predicted_ratings)
