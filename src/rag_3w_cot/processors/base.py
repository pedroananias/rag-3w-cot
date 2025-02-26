import asyncio
import hashlib
import importlib
from functools import cached_property
from pathlib import Path
from typing import Any, List

import pandas as pd
from loguru import logger
from pydantic import BaseModel

from rag_3w_cot.embeddings import EmbeddingsFactory
from rag_3w_cot.settings import Settings
from rag_3w_cot.vectorstores import BaseVectorStore


class BaseProcessor(BaseModel):
    settings: Settings
    data_path: Path
    metadata_file: Path

    @property
    def enable_cache(self) -> bool:
        return self.settings.processing_enable_cache

    @property
    def force_gpu_cache_release(self) -> bool:
        return self.settings.force_gpu_cache_release

    @property
    def max_concurrent_tasks(self) -> int:
        return self.settings.processing_max_concurrent_tasks

    @cached_property
    def vectorstore(self) -> BaseVectorStore:
        embeddings = EmbeddingsFactory.create_model(self.settings)

        vectorstore_cls = getattr(
            importlib.import_module("rag_3w_cot.vectorstores"),
            self.settings.processing_vectorstore,
        )

        vectorstore_path = self.get_vectorstore_cache_path(self.data_path)

        return vectorstore_cls(
            settings=self.settings, embeddings=embeddings, index_path=vectorstore_path
        )

    @cached_property
    def available_files(self) -> List[Path]:
        return [
            file
            for file in self.data_path.glob("*")
            if file.suffix in self.settings.processing_allowed_extensions
        ]

    @cached_property
    def df_metadata(self) -> pd.DataFrame:
        try:
            df = pd.read_json(self.metadata_file)
        except ValueError:
            df = pd.read_csv(self.metadata_file)

        available_sha1 = [file.stem for file in self.available_files]

        return pd.DataFrame(
            df[df["sha1"].isin(available_sha1)][["sha1", "company_name"]]
        )

    @cached_property
    def vectorstore_cache_hash(self) -> str:
        hash_components = (
            self.settings.processing_vectorstore,
            self.settings.processing_document_html_to_markdown,
            self.settings.processing_decument_deduplicate,
            self.settings.processing_document_filter_similar_documents_threshold,
            self.settings.processing_document_filter_small_documents_chars,
            self.settings.embeddings_model,
            self.settings.embeddings_huggingface_precision,
            self.settings.embeddings_huggingface_batch_size,
        )
        combined_string = "".join(str(component) for component in hash_components)
        return hashlib.md5(combined_string.encode()).hexdigest()

    @cached_property
    def json_cache_hash(self) -> str:
        hash_components = (
            self.settings.llm_chunk_size,
            self.settings.unstructured_strategy,
            self.settings.unstructured_chunking_strategy,
            self.settings.unstructured_multipage_sections,
            self.settings.unstructured_overlap,
        )
        combined_string = "".join(str(component) for component in hash_components)
        return hashlib.md5(combined_string.encode()).hexdigest()

    def process(self, *args, **kwargs) -> Any:
        return asyncio.run(self.async_process(*args, **kwargs))

    async def async_process(self, *args, **kwargs):
        raise NotImplementedError()

    def cache_path(self, path: Path, type_: str) -> Path:
        cache_dir = (path if path.is_dir() else path.parent) / ".cache"
        cache_hash = getattr(self, f"{type_}_cache_hash")
        cache_path: Path = (
            cache_dir / f".{type_}" / cache_hash / path.with_suffix(f".{type_}").name
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        return cache_path

    def get_vectorstore_cache_path(self, path: Path) -> Path:
        return self.cache_path(path, "vectorstore")

    def get_json_cache_path(self, path: Path) -> Path:
        return self.cache_path(path, "json")

    async def cleanup_if_no_cache(self, path: Path):
        if self.enable_cache:
            return

        logger.error(f"Cache is disabled, cleaning up {path}'s cache...")

        for extension in ["faiss", "json"]:
            cache_path = self.cache_path(path, extension)

            if cache_path.is_dir():
                tasks = []
                for f in cache_path.glob("*"):
                    print(f"Removing {f}")
                    tasks.append(asyncio.to_thread(f.unlink))  # Run unlink in a thread

                await asyncio.gather(*tasks)  # Run deletions concurrently
                await asyncio.to_thread(cache_path.rmdir)
                print(f"Removing {cache_path}")
            else:
                print(f"Removing {cache_path}")
                await asyncio.to_thread(cache_path.unlink, missing_ok=True)
