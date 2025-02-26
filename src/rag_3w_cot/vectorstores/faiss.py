from functools import cached_property
from typing import List, Optional
from uuid import uuid4

import faiss
import torch
from langchain_community.vectorstores import FAISS, DistanceStrategy
from langchain_core.documents import Document as LangChainDocument
from loguru import logger

from rag_3w_cot.models import Document
from rag_3w_cot.vectorstores import BaseVectorStore


class FAISSVectorStore(BaseVectorStore):
    @cached_property
    def vectorstore(self) -> FAISS:
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Cached FAISS vectorstore not found: {self.index_path}"
            )

        logger.debug(f"Loading FAISS vectorstore from {self.index_path}...")

        vectorstore = FAISS.load_local(
            str(self.index_path),
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True,
        )

        vectorstore.index = faiss.index_gpu_to_cpu(vectorstore.index)
        return vectorstore

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        faiss.omp_set_num_threads(self.max_concurrent_tasks)

    def create(self, documents: List[Document]) -> FAISS:
        langchain_documents = self.to_vectorstore_documents(documents)

        with torch.no_grad():
            vectorstore = FAISS.from_documents(
                langchain_documents,
                embedding=self.embeddings,
                distance_strategy=DistanceStrategy.COSINE,
            )

        vectorstore.save_local(str(self.index_path))
        return vectorstore

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
        retriever = self.vectorstore.as_retriever(
            search_type=search_type, search_kwargs=kwargs
        )
        return await retriever.ainvoke(question, verbose=False)
