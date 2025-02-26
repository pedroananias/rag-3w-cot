# rag-3w-cot

## Summary

A 3-way FAISS MMR Search & Stepped Chain Of Thought RAG.

It uses a local deployed Unstructured API (open source) for PDF extraction (high resolution) and chunking, Hugging Face's pipeline for models.

The `3-way FAISS MMR Search` comprehends of:

1. expanding the query with dictionary terms (e.g. financial)
2. selecting related files (exact matches + cosine similarity)
3. searching the FAISS vetor database using the maximum marginal relevance (MMR, top_k) for both texts and Markdown (pre-converted from HTML) contents (e.g. tables).

The `Stepped Chain Of Thought` pipeline chains 3 calls (`input` -> `output`) to the model:

1. **reasoning**: `Query` & `Document` selection, reasoning & answering tasks
2. **formatting**: number scaling (e.g., `122k` -> `122000.0`, `122 million` -> `122000000.0`) & formatting tasks
3. **parsing**: `Answer` JSON parsed object output task

## PDF Data Extraction

The algorithm relies on [Unstructured API](https://github.com/Unstructured-IO/unstructured-api/) for PDF data extraction. The endpoint can be deployed using one of the following approaches:

1. Local installation (with GPU support): [Developer Quick Start](https://github.com/Unstructured-IO/unstructured-api/)
2. Docker container (without GPU support):

```bash
docker run -p 9500:9500 -d --rm --name unstructured-api -e PORT=9500 downloads.unstructured.io/unstructured-io/unstructured-api:latest
```

### Strategies

There are two main strategies that can be used for PDF data extraction when calling [Unstructured API](https://github.com/Unstructured-IO/unstructured-api/):

1. `fast`: better suited for documents without image-embedded text, but will certainly degrade answer quality
2. `hi_res`: almost 20x times slower, but provides higher precision with the ability to extract tables as HTML

## Quick Start

This is a [Poetry](https://python-poetry.org/) project. You may quickly experiment by running:

1. `copy .env.example .env`
2. `make install`
3. `make download_language_models`
4. `poetry run python samples/rag_3w_cot_pipeline.py`

## Architecture

- `dictionaries`: dictionary terms (e.g. financial) wrapped by `BaseTermsDictionary`
- `embeddings`: local (Hugging Face's repositories) or OpenAI (LangChain) registered via `EmbeddingsFactory`
- `evaluations`: a set of evaluation metrics (e.g. exact match, cosine similarity, rouge score) inherited from `BaseEvaluation`
- `llms`: local (Hugging Face's repositories) or OpenAI (LangChain) models inherited from `BaseLLM`
- `models`: the `Query`, `Document` & `Answer` models
- `pipelines`: custom pipelines (e.g. CoT) inherited from `BasePipeline`
- `processors`: `Document` (e.g. extraction, cleaning) & `Query` (e.g. vector store searches) processors inherited from `BaseProcessor`
- `prompts`: a set of Markdown prompts wrapped by `BasePrompt`
- `vectorstores`: `FAISSVectorStore` & `EnsembleFAISSBM25VectorStore` vector stores inherited from `BaseVectorStore`
- `settings.py`: shared `Settings` model aggregating all the algorithm's parameteres
- `utils.py`: set of utilities functions
