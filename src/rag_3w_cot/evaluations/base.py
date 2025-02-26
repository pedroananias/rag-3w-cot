from typing import Any, List, Tuple

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel

from rag_3w_cot.embeddings import EmbeddingsFactory
from rag_3w_cot.models import Answer
from rag_3w_cot.settings import Settings
from rag_3w_cot.utils import (
    force_gpu_cache_release,
)


class BaseEvaluation(BaseModel):
    settings: Settings

    answers: List[Answer]
    true_answers: List[Answer]

    @property
    def embeddings(self) -> Embeddings:
        return EmbeddingsFactory.create_model(self.settings)

    @property
    def parsed_data(self) -> List[Tuple[str, Any, Any]]:
        evaluation_data = []
        for true_answer in self.true_answers:
            true_answer_string = str(true_answer.value or "").strip().lower()
            filtered_answers = list(
                filter(
                    lambda a: a.question_text == true_answer.question_text, self.answers
                )
            )

            if filtered_answers:
                answer_string = str(filtered_answers[0].value or "").strip().lower()
                evaluation_data.append(
                    (true_answer.question_text, answer_string, true_answer_string)
                )
            else:
                evaluation_data.append(
                    (true_answer.question_text, None, true_answer_string)
                )

        return evaluation_data

    @property
    def _answer_true_answer_pairs(self) -> List[Tuple[str, str]]:
        return [(str(data[1]), str(data[2])) for data in self.parsed_data]

    def __init__(self, **data):
        super().__init__(**data)

        if self.settings.force_gpu_cache_release:
            force_gpu_cache_release()

    def get_score(self) -> float:
        raise NotImplementedError()
