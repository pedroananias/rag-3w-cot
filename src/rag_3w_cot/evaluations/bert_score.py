import bert_score

from rag_3w_cot.utils import (
    force_gpu_cache_release,
)

from .base import BaseEvaluation


class BERTScoreEvaluation(BaseEvaluation):
    def get_score(self) -> float:
        if self.settings.force_gpu_cache_release:
            force_gpu_cache_release()

        answers_strings = [answer for answer, _ in self._answer_true_answer_pairs]
        true_answers_strings = [
            true_answer for _, true_answer in self._answer_true_answer_pairs
        ]

        *_, F1 = bert_score.score(answers_strings, true_answers_strings, lang="en")
        return F1.mean().item()  # pyright: ignore
