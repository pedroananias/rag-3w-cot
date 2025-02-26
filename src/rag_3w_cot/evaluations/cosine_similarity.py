import numpy as np

from rag_3w_cot.utils import get_cosine_similarity

from .base import BaseEvaluation


class CosineSimilarityEvaluation(BaseEvaluation):
    def get_score(self) -> float:
        similarities = []
        for answer, true_answer in self._answer_true_answer_pairs:
            similarities.append(
                get_cosine_similarity(answer, true_answer, stop_words=False)
            )

        return float(np.mean(similarities or 0.0))
