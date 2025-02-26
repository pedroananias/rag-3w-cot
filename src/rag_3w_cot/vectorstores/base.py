import asyncio
from functools import cached_property
from pathlib import Path
from typing import Any, List, Optional

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict

from rag_3w_cot.models import Document
from rag_3w_cot.settings import Settings
from rag_3w_cot.utils import get_cosine_similarity


class BaseVectorStore(BaseModel):
    settings: Settings
    embeddings: Embeddings
    index_path: Path

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    def max_concurrent_tasks(self) -> int:
        return self.settings.processing_max_concurrent_tasks

    @property
    def document_text_score_threshold(self) -> float:
        return self.settings.processing_query_similarity_document_text_score_threshold

    @property
    def document_text_lambda_mult(self) -> float:
        return self.settings.processing_query_similarity_document_text_lambda_mult

    @property
    def document_text_top_k(self) -> int:
        return self.settings.processing_query_similarity_document_text_top_k

    @property
    def document_type_score_threshold(self) -> float:
        return self.settings.processing_query_similarity_document_type_score_threshold

    @property
    def document_type_lambda_mult(self) -> float:
        return self.settings.processing_query_similarity_document_type_lambda_mult

    @property
    def document_type_top_k(self) -> int:
        return self.settings.processing_query_similarity_document_type_top_k

    @cached_property
    def vectorstore(self) -> Any:
        raise NotImplementedError()

    def create(self, documents: List[Document]) -> Any:
        raise NotImplementedError()

    async def async_similarity_search(self, *args, **kwargs) -> List[Any]:
        raise NotImplementedError()

    async def async_mmr_search(self, *args, **kwargs) -> List[Any]:
        raise NotImplementedError()

    async def async_search(
        self,
        question: str,
        filter: Optional[dict] = None,
        type_: str = "similarity_search",
    ) -> List[Document]:
        content_type = "text" if not filter or "type" not in filter else filter["type"]
        content_type = "type" if content_type != "text" else "text"

        top_k = getattr(self, f"document_{content_type}_top_k")
        score_threshold = getattr(self, f"document_{content_type}_score_threshold")
        lambda_mult = getattr(self, f"document_{content_type}_lambda_mult")

        vectorstore_documents = []
        match type_:
            case "similarity_search":
                vectorstore_documents = await self.async_similarity_search(
                    question,
                    top_k=top_k,
                    score_threshold=score_threshold,
                    lambda_mult=lambda_mult,
                    filter=filter,
                )
            case "mmr_search":
                vectorstore_documents = await self.async_mmr_search(
                    question,
                    top_k=top_k,
                    lambda_mult=lambda_mult,
                    filter=filter,
                )
            case _:
                raise ValueError(f"Unknown search type: {type_}")

        documents = self.from_vectorstore_documents(vectorstore_documents)
        documents = self.deduplicate(documents)
        documents = self.filter_documents(documents, filter)
        documents = self.add_scores(question, documents)
        documents = self.sort_by_score(documents)

        return documents

    def similarity_search(self, *args, **kwargs) -> List[Any]:
        return asyncio.run(self.async_similarity_search(*args, **kwargs))

    def mmr_search(self, *args, **kwargs) -> List[Any]:
        return asyncio.run(self.async_mmr_search(*args, **kwargs))

    def search(
        self,
        question: str,
        filter: Optional[dict] = None,
        type_: str = "similarity_search",
    ) -> List[Document]:
        return asyncio.run(
            self.async_search(question=question, filter=filter, type_=type_)
        )

    def deduplicate(self, documents: List[Document]) -> List[Document]:
        deduplicated_documents = {}
        for document in documents:
            deduplicated_documents[document.id] = document

        return list(deduplicated_documents.values())

    def add_scores(self, question: str, documents: List[Document]) -> List[Document]:
        documents_with_score = []
        for document in documents:
            document.metadata["score"] = get_cosine_similarity(
                question, document.page_content
            )
            documents_with_score.append(document)

        return documents_with_score

    def sort_by_score(self, documents: List[Document]) -> List[Document]:
        if not documents or len(documents) <= 1:
            return documents or []

        return sorted(
            documents,
            key=lambda document: document.metadata.get("score", 0.0),
            reverse=True,
        )

    def to_vectorstore_documents(self, documents: List[Document]) -> List[Any]:
        raise NotImplementedError()

    def from_vectorstore_documents(self, other_documents: List[Any]) -> List[Document]:
        raise NotImplementedError()

    def filter_documents(
        self, documents: List[Document], filter: Optional[dict]
    ) -> List[Document]:
        filtered_documents = []
        for document in documents:
            filter_match = (
                True
                if not filter
                else all(
                    document.metadata.get(key) == value for key, value in filter.items()
                )
            )
            if filter_match:
                filtered_documents.append(document)

        return filtered_documents
