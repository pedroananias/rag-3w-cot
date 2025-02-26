import evaluate
import numpy as np

from .base import BaseEvaluation


class RougeScoreEvaluation(BaseEvaluation):
    def get_score(self):
        answers_strings = [answer for answer, _ in self._answer_true_answer_pairs]
        true_answers_strings = [
            true_answer for _, true_answer in self._answer_true_answer_pairs
        ]

        rouge = evaluate.load("rouge")
        scores = rouge.compute(
            predictions=answers_strings, references=true_answers_strings
        )
        return float(np.mean([score for score in (scores or {}).values()])) or 0.0
