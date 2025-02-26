import numpy as np

from rag_3w_cot.utils import get_vector_cosine_similarity

from .base import BaseEvaluation


class EmbeddingCosineSimilarityEvaluation(BaseEvaluation):
    def get_score(self) -> float:
        similarities = []
        for answer, true_answer in self._answer_true_answer_pairs:
            answer_embedding = self.embeddings.embed_query(answer)
            true_answer_embedding = self.embeddings.embed_query(true_answer)
            similarities.append(
                get_vector_cosine_similarity(answer_embedding, true_answer_embedding)
            )

        return float(np.mean(similarities or 0.0))
