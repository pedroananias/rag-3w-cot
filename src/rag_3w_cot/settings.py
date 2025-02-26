from pathlib import Path
from typing import List, Literal, Optional

import torch
from loguru import logger
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    force_gpu_cache_release: bool = True

    llm: Literal[
        "DeepSeekR1Llama8B",
        "MicrosoftPhi4",
        "MicrosoftPhi4Mini",
        "Qwen257B",
        "Qwen2514B",
        "OpenAIChatGPT",
    ] = "MicrosoftPhi4Mini"
    llm_chunk_size: int = 1536
    llm_batch_size: int = 4
    llm_temperature: float = 0.0
    llm_pipeline_top_p: Optional[float] = None
    llm_repetition_penalty: Optional[float] = None
    llm_padding: bool = True
    llm_num_beams: int = 1
    llm_early_stopping: bool = False
    llm_do_sample: bool = False
    llm_use_fast: bool = True
    llm_padding: bool = True
    llm_flash_attention_2: bool = True
    llm_quantization_type: Literal["fp16", "int8", "int4"] = "fp16"

    embeddings_model: Literal[
        "BAAI/bge-large-en",
        "sentence-transformers/all-MiniLM-L12-v2",
        "openai/text-embedding-3-small",
    ] = "BAAI/bge-large-en"
    embeddings_huggingface_precision: Literal["float32", "int8"] = "float32"
    embeddings_huggingface_batch_size: int = 4

    processing_enable_cache: bool = True
    processing_max_concurrent_tasks: int = 4
    processing_allowed_extensions: List[str] = [".pdf"]
    processing_use_normalized_query: bool = False
    processing_vectorstore: Literal[
        "FAISSVectorStore", "EnsembleFAISSBM25VectorStore"
    ] = "FAISSVectorStore"
    processing_query_terms_dictionary: List[str] | None = ["FinancialTermsDictionary"]
    processing_query_search_type: Literal["similarity_search", "mmr_search"] = (
        "mmr_search"
    )
    processing_query_similarity_file_score_threshold: float = 0.5
    processing_query_similarity_document_text_score_threshold: float = 0.50
    processing_query_similarity_document_text_lambda_mult: float = 1.0
    processing_query_similarity_document_text_top_k: int = 20
    processing_query_similarity_document_type_score_threshold: float = 0.25
    processing_query_similarity_document_type_lambda_mult: float = 1.0
    processing_query_similarity_document_type_top_k: int = 10
    processing_document_html_to_markdown: bool = True
    processing_decument_deduplicate: bool = True
    processing_document_filter_similar_documents_threshold: float | None = 0.95
    processing_document_filter_small_documents_chars: int | None = 200

    # see: https://github.com/Unstructured-IO/unstructured-api
    unstructured_url: str = "http://localhost:9500/general/v0/general"
    unstructured_strategy: Literal["auto", "hi_res", "fast"] = "hi_res"
    unstructured_chunking_strategy: Literal["basic", "by_title"] = "by_title"
    unstructured_multipage_sections: bool = True
    unstructured_overlap: int = 0

    open_api_key: str = Field(alias="OPENAI_API_KEY")
    huggingface_api_key: str = Field(alias="HF_TOKEN")
    unstructured_api_key: Optional[str] = Field(
        alias="UNSTRUCTURED_API_KEY", default=None
    )

    model_config = SettingsConfigDict(
        env_file=".env",
    )

    def export(self, file: Path):
        file.write_text(self.dump_json())

    def dump_json(self) -> str:
        forbidden_keys = {key for key in self.model_fields if key.endswith("_key")}
        return self.model_dump_json(exclude=forbidden_keys, indent=4)

    def debug(self):
        logger.debug(self.dump_json())
