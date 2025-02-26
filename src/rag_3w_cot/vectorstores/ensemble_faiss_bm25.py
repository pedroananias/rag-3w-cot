import pickle
from functools import cached_property
from typing import List, Optional, Tuple
from uuid import uuid4

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document as LangChainDocument
from loguru import logger

from rag_3w_cot.models import Document
from rag_3w_cot.vectorstores.base import BaseVectorStore
from rag_3w_cot.vectorstores.faiss import FAISSVectorStore


class EnsembleFAISSBM25VectorStore(BaseVectorStore):
    weights: List[float] = [0.75, 0.25]

    @cached_property
    def vectorstore(self) -> Tuple[FAISS, BM25Retriever]:
        bm25_retriever_path = self.index_path / "bm25_retriever.pkl"

        if not bm25_retriever_path.exists():
            raise FileNotFoundError(
                f"Cached BM25 retriever not found: {self.index_path}"
            )

        logger.debug(
            f"Loading FAISS vectorstore & BM25 retriever from {self.index_path}..."
        )

        with open(bm25_retriever_path, "rb") as f:
            bm25_retriever = pickle.load(f)

        vectorstore = FAISSVectorStore(
            settings=self.settings,
            embeddings=self.embeddings,
            index_path=self.index_path,
        ).vectorstore

        return vectorstore, bm25_retriever

    def create(self, documents: List[Document]) -> Tuple[FAISS, BM25Retriever]:
        vectorstore = FAISSVectorStore(
            settings=self.settings,
            embeddings=self.embeddings,
            index_path=self.index_path,
        ).create(documents)
        langchain_documents = self.to_vectorstore_documents(documents)

        bm25_retriever = BM25Retriever.from_documents(langchain_documents)
        bm25_retriever_path = self.index_path / "bm25_retriever.pkl"

        with open(bm25_retriever_path, "wb") as f:
            pickle.dump(bm25_retriever, f)

        return vectorstore, bm25_retriever

    async def async_similarity_search(
        self,
        question: str,
        top_k: int = 4,
        score_threshold: float = 0.6,
        lambda_mult: float = 0.0,
        filter: Optional[dict] = None,
    ) -> List[LangChainDocument]:
        return await self._async_search_helper(
            question=question,
            search_type="similarity_score_threshold",
            k=top_k,
            score_threshold=score_threshold,
            lambda_mult=lambda_mult,
            filter=filter,
        )

    async def async_mmr_search(
        self,
        question: str,
        top_k: int = 4,
        lambda_mult: float = 0.0,
        filter: Optional[dict] = None,
    ) -> List[LangChainDocument]:
        return await self._async_search_helper(
            question=question,
            search_type="mmr",
            k=top_k,
            lambda_mult=lambda_mult,
            filter=filter,
        )

    def from_vectorstore_documents(
        self, other_documents: List[LangChainDocument]
    ) -> List[Document]:
        return [
            Document(
                id=doc.id or str(uuid4()),
                page_content=doc.page_content,
                metadata=doc.metadata,
            )
            for doc in other_documents
        ]

    def to_vectorstore_documents(
        self, documents: List[Document]
    ) -> List[LangChainDocument]:
        return [
            LangChainDocument(
                id=doc.id,
                page_content=doc.page_content,
                metadata=doc.metadata,
            )
            for doc in documents
        ]

    async def _async_search_helper(
        self, question: str, search_type: str, **kwargs
    ) -> List[LangChainDocument]:
        vectorstore, bm25_retriever = self.vectorstore

        faiss_retriever = vectorstore.as_retriever(
            search_type=search_type, search_kwargs=kwargs
        )

        ensemble_retriever = EnsembleRetriever(
            retrievers=[faiss_retriever, bm25_retriever],
            weights=self.weights,
        )

        return await ensemble_retriever.ainvoke(question, verbose=False)
