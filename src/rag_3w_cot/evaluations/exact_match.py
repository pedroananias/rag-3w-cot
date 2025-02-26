import numpy as np

from .base import BaseEvaluation


class ExactMatchEvaluation(BaseEvaluation):
    def get_score(self) -> float:
        scores = [
            1.0 if answer == true_answer else 0.0
            for answer, true_answer in self._answer_true_answer_pairs
        ]
        return float(np.mean(scores or 0.0))
