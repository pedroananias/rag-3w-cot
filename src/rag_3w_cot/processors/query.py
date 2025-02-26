import asyncio
import importlib
import itertools
from pathlib import Path
from typing import List, Set, Type

from loguru import logger

from rag_3w_cot.dictionaries import BaseTermsDictionary
from rag_3w_cot.models import Document, Query
from rag_3w_cot.utils import force_gpu_cache_release, get_cosine_similarity

from .base import BaseProcessor


class QueryProcessor(BaseProcessor):
    @property
    def query_search_type(self) -> str:
        return self.settings.processing_query_search_type

    @property
    def use_normalized_query(self) -> bool:
        return self.settings.processing_use_normalized_query

    @property
    def file_score_threshold(self) -> float:
        return self.settings.processing_query_similarity_file_score_threshold

    @property
    def deduplicate(self) -> bool:
        return self.settings.processing_decument_deduplicate

    @property
    def filter_similar_documents_threshold(self) -> float | None:
        return self.settings.processing_document_filter_similar_documents_threshold

    @property
    def filter_small_documents_chars(self) -> int | None:
        return self.settings.processing_document_filter_small_documents_chars

    @property
    def html_to_markdown(self) -> bool:
        return self.settings.processing_document_html_to_markdown

    @property
    def query_terms_dictionary(self) -> List[Type[BaseTermsDictionary]] | None:
        if not self.settings.processing_query_terms_dictionary:
            return None

        dictionary_cls = []
        for dictionary_name in self.settings.processing_query_terms_dictionary:
            dictionary_cls.append(
                getattr(
                    importlib.import_module("rag_3w_cot.dictionaries"), dictionary_name
                )
            )

        return dictionary_cls

    async def async_process(self, queries: List[Query]) -> List[Query]:
        logger.success(f"{len(queries)} queries found!")

        tasks = [self._process_single_query(query) for query in queries]
        queries = await asyncio.gather(*tasks)

        return queries

    async def _process_single_query(self, query: Query) -> Query:
        logger.warning(f"{query.question_text}: processing...")

        if self.force_gpu_cache_release:
            force_gpu_cache_release()

        relevant_files = await self._get_relevant_files(query)
        query.set_relevant_files(relevant_files)

        relevant_documents = await self._get_relevant_documents(query, relevant_files)
        query.set_relevant_documents(relevant_documents)

        logger.success(f"{query.question_text}: processed!")

        return query

    async def _get_relevant_files(self, query: Query) -> Set[Path]:
        available_owners = self.df_metadata["company_name"].tolist()
        available_sha1 = self.df_metadata["sha1"].tolist()

        exact_matched_owners = {
            owner for owner in available_owners if owner in query.question_text
        }

        async def _compute_similarity(query_text: str, owner: str) -> float:
            return await asyncio.to_thread(get_cosine_similarity, query_text, owner)

        similarity_tasks = [
            _compute_similarity(query.question_text, owner)
            for owner in available_owners
        ]
        similarity_scores = await asyncio.gather(*similarity_tasks)
        similarity_matached_owners = {
            owner
            for owner, score in zip(available_owners, similarity_scores)
            if score >= self.file_score_threshold
        }

        matched_owners = exact_matched_owners | similarity_matached_owners
        matched_sha1 = self.df_metadata[
            self.df_metadata["company_name"].isin(list(matched_owners))
        ]["sha1"].tolist()

        logger.debug(
            f"{query.question_text}: matched owners {matched_owners} & sha1 {matched_sha1}"
        )

        files = set()
        for sha1 in matched_sha1 or available_sha1:
            files.add(self.data_path / f"{sha1}.pdf")

        return files

    async def _get_relevant_documents(
        self, query: Query, files: Set[Path]
    ) -> List[Document]:
        documents = []

        question = self._get_question_expanded(query)

        tasks = [
            self._get_relevant_documents_per_file(question, file) for file in files
        ]
        results = await asyncio.gather(*tasks)
        documents = list(itertools.chain.from_iterable(results))

        return self.vectorstore.sort_by_score(documents)

    async def _get_relevant_documents_per_file(
        self, question: str, file: Path
    ) -> List[Document]:
        text_relevant_documents = await self.vectorstore.async_search(
            question=question,
            filter={"pdf_sha1": file.stem, "content_type": "text"},
            type_=self.query_search_type,
        )
        if not text_relevant_documents:
            return []

        type_relevant_documents = await self.vectorstore.async_search(
            question=question,
            filter={
                "pdf_sha1": file.stem,
                "content_type": "markdown" if self.html_to_markdown else "html",
            },
            type_=self.query_search_type,
        )

        return text_relevant_documents + type_relevant_documents

    def _get_question_expanded(self, query: Query) -> str:
        if self.query_terms_dictionary:
            query.set_dicionaries(self.query_terms_dictionary)
            question = query.question_expanded
            if self.use_normalized_query:
                question = query.question_normalized_expanded
        else:
            question = query.question_text

        return question
