from typing import Callable, Dict

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from rag_3w_cot.settings import Settings


class EmbeddingsFactory:
    registry: Dict[str, Callable] = {}

    @classmethod
    def register(cls, model, creator: Callable[[Settings], Embeddings]):
        cls.registry[model] = creator

    @classmethod
    def create_model(cls, settings: Settings) -> Embeddings:
        try:
            return cls.registry[settings.embeddings_model](settings)
        except KeyError:
            return cls.registry["huggingface"](settings)


EmbeddingsFactory.register(
    "huggingface",
    lambda settings: HuggingFaceEmbeddings(
        model_name=settings.embeddings_model,
        model_kwargs={"device": settings.device},
        encode_kwargs={
            "normalize_embeddings": True,
            "precision": settings.embeddings_huggingface_precision,
            "batch_size": settings.embeddings_huggingface_batch_size,
        },
    ),
)

EmbeddingsFactory.register(
    "openai/text-embedding-3-small",
    lambda settings: OpenAIEmbeddings(
        api_key=settings.open_api_key,  # pyright: ignore
        model="text-embedding-3-small",
        chunk_size=settings.llm_chunk_size,
        retry_min_seconds=10,
    ),
)

EmbeddingsFactory.register(
    "openai/text-embedding-3-large",
    lambda settings: OpenAIEmbeddings(
        api_key=settings.open_api_key,  # pyright: ignore
        model="text-embedding-3-large",
        chunk_size=settings.llm_chunk_size,
        retry_min_seconds=10,
    ),
)
