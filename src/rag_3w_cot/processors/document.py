import asyncio
import itertools
import json
import re
from pathlib import Path
from typing import List, Tuple

import aiohttp
from loguru import logger
from markdownify import markdownify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag_3w_cot.models import Document
from rag_3w_cot.utils import (
    extract_most_common_year,
    force_gpu_cache_release,
    normalize_sentence,
)

from .base import BaseProcessor


class DocumentProcessor(BaseProcessor):
    @property
    def llm_chunk_size(self) -> int:
        return self.settings.llm_chunk_size

    @property
    def unstructured_url(self) -> str:
        return self.settings.unstructured_url

    @property
    def unstructured_strategy(self) -> str:
        return self.settings.unstructured_strategy

    @property
    def unstructured_chunking_strategy(self) -> str:
        return self.settings.unstructured_chunking_strategy

    @property
    def unstructured_overlap(self) -> int:
        return self.settings.unstructured_overlap

    @property
    def unstructured_multipage_sections(self) -> bool:
        return self.settings.unstructured_multipage_sections

    @property
    def html_to_markdown(self) -> bool:
        return self.settings.processing_document_html_to_markdown

    @property
    def deduplicate(self) -> bool:
        return self.settings.processing_decument_deduplicate

    @property
    def filter_small_documents_chars(self) -> int | None:
        return self.settings.processing_document_filter_small_documents_chars

    @property
    def filter_similar_documents_threshold(self) -> float | None:
        return self.settings.processing_document_filter_similar_documents_threshold

    @property
    def _asyncio_semaphore(self) -> asyncio.Semaphore:
        return asyncio.Semaphore(self.settings.processing_max_concurrent_tasks)

    async def async_process(self):
        logger.success(f"{len(self.available_files)} files found!")

        vectorstore_path = self.get_vectorstore_cache_path(self.data_path)
        if self.enable_cache and vectorstore_path.exists():
            logger.warning(f"{self.data_path}: loading cached vectorstore...")
            return

        tasks = [self._process_single_document(file) for file in self.available_files]
        documents = list(itertools.chain(*await asyncio.gather(*tasks)))

        await self.cleanup_if_no_cache(self.data_path)

        logger.warning(f"{self.data_path}: (re)creating vectorestore...")
        self.vectorstore.create(documents)

    async def _process_single_document(self, file: Path) -> List[Document]:
        async with self._asyncio_semaphore:
            logger.warning(f"{file}: processing...")

            if self.force_gpu_cache_release:
                force_gpu_cache_release()

            await self.cleanup_if_no_cache(file)

            try:
                if not self.enable_cache:
                    raise FileNotFoundError
                documents = await self._load_cached_documents(file)
            except FileNotFoundError:
                logger.warning(f"{file}: cache not found, calling Unstructured API...")

                documents = await self._call_unstructured(file)
                await self._cache_documents(documents, file)

            logger.debug(f"{file}: {len(documents)} document(s) found")

            if self.html_to_markdown:
                documents, total_converted = await self._html_to_markdown(documents)
                logger.debug(
                    f"{file}: {total_converted} document(s) converted to markdown"
                )

            if self.deduplicate:
                documents = await self._deduplicate_documents(documents)
                logger.debug(
                    f"{file}: {len(documents)} document(s) after deduplication"
                )

            if self.filter_small_documents_chars:
                documents = await asyncio.to_thread(
                    self._filter_small_documents, documents
                )
                logger.debug(
                    f"{file}: {len(documents)} document(s) after size filtering"
                )

            if self.filter_similar_documents_threshold:
                documents = await self._filter_similar_documents(documents)
                logger.debug(
                    f"{file}: {len(documents)} document(s) after similarity filtering"
                )

            logger.success(f"{file}: processed!")

        return documents

    async def _call_unstructured(self, file: Path) -> List[Document]:
        payload = {
            "filename": file.name,
            "response_type": "application/json",
            "coordinates": False,
            "encoding": "utf-8",
            "strategy": self.unstructured_strategy,
            "hi_res_model_name": None,
            "include_page_breaks": False,
            "skip_infer_table_types": [],
            "xml_keep_tags": False,
            "languages": "eng",
            "extract_image_block_types": None,
            "unique_element_ids": True,
            "chunking_strategy": self.unstructured_chunking_strategy,
            "combine_under_n_chars": self.llm_chunk_size,
            "max_characters": self.llm_chunk_size,
            "multipage_sections": self.unstructured_multipage_sections,
            "new_after_n_chars": None,
            "overlap": self.unstructured_overlap,
            "overlap_all": False,
            "starting_page_number": None,
            "include_slide_notes": True,
            "split_pdf_page": False,
            "files": (str(file), await asyncio.to_thread(file.read_bytes)),
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.settings.unstructured_url,
                data=payload,
            ) as response:
                if response.status != 200:
                    raise ConnectionError(
                        f"Error processing file {file}: {await response.text()}"
                    )

                response_json = await response.json()

        return await self._json_to_documents(response_json)

    async def _json_to_documents(self, unstructured_data: List[dict]) -> List[Document]:
        if not unstructured_data:
            return []

        most_common_year = await asyncio.to_thread(
            extract_most_common_year,
            unstructured_data,
            content_key="text" if "text" in unstructured_data[0] else "page_content",
        )

        documents = []
        for item in unstructured_data:
            content = str(item.get("text", item.get("page_content")))
            if not content:
                continue

            id_ = item.get("id", item.get("element_id"))
            metadata = dict(item.get("metadata", {}))
            filename = str(metadata.get("filename", metadata.get("pdf_sha1", "")))
            sha1 = filename.split(".")[0]
            page_index = metadata.get(
                "page_index", int(metadata.get("page_number", 0)) - 1
            )

            try:
                owner = self.df_metadata[self.df_metadata["sha1"] == sha1].iloc[0][
                    "company_name"
                ]
            except IndexError:
                owner = filename

            content_type = metadata.get("content_type", "text")
            if "text_as_html" in metadata:
                content_type = "html"
                content = metadata.get("text_as_html")

            document = Document(
                id=str(id_),
                page_content=str(content),
                metadata={
                    "pdf_sha1": str(sha1),
                    "page_index": int(page_index),
                    "owner": str(owner),
                    "year": int(most_common_year),
                    "content_type": content_type,
                },
            )

            documents.append(document)

        return documents

    async def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        page_content_seen = set()
        deduplicated = []
        for document in documents:
            normalized_content = await asyncio.to_thread(
                normalize_sentence, document.page_content
            )
            if normalized_content not in page_content_seen:
                page_content_seen.add(normalized_content)
                deduplicated.append(document)

        return deduplicated

    async def _filter_similar_documents(
        self, documents: List[Document]
    ) -> List[Document]:
        if (
            not documents
            or len(documents) <= 1
            or not self.filter_similar_documents_threshold
        ):
            return documents or []

        vectorizer = TfidfVectorizer(stop_words="english")
        doc_texts = [doc.page_content for doc in documents]
        tfidf_matrix = await asyncio.to_thread(vectorizer.fit_transform, doc_texts)
        similarity_matrix = await asyncio.to_thread(cosine_similarity, tfidf_matrix)

        to_remove = set()
        for i in range(len(documents)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(documents)):
                if j in to_remove:
                    continue
                if similarity_matrix[i, j] > self.filter_similar_documents_threshold:
                    to_remove.add(j)

        return [doc for idx, doc in enumerate(documents) if idx not in to_remove]

    def _filter_small_documents(self, documents: List[Document]) -> List[Document]:
        filtered_documents = [
            document
            for document in documents
            if len(document.page_content) > (self.filter_small_documents_chars or 0)
        ]

        return filtered_documents

    async def _html_to_markdown(
        self, documents: List[Document]
    ) -> Tuple[List[Document], int]:
        total_converted = 0
        converted_documents = []
        for document in documents:
            if not document.metadata.get("content_type") == "html":
                converted_documents.append(document)
                continue

            markdown_content = await asyncio.to_thread(
                markdownify, html=document.page_content
            )
            markdown_content = markdown_content.replace("\xa0", " ").strip()

            document.metadata["content_type"] = "markdown"
            document.page_content = re.sub(r"\n\s*\n", "\n\n", markdown_content)
            converted_documents.append(document)

            total_converted += 1

        return converted_documents, total_converted

    async def _cache_documents(self, documents: List[Document], output_file: Path):
        output_file = self.get_json_cache_path(output_file)

        dumped_documents = [d.model_dump() for d in documents]
        json_string = json.dumps(dumped_documents, indent=4)

        await asyncio.to_thread(output_file.write_text, json_string)

    async def _load_cached_documents(self, file: Path) -> List[Document]:
        file = self.get_json_cache_path(file)
        if not file.exists():
            raise FileNotFoundError(f"Cache file {file} not found")

        return await self._json_to_documents(json.loads(file.read_text()))
