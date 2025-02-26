SHELL := /bin/bash

clean:
	find . | grep "__pycache__" | xargs rm -rf
	rm -rf dist/ build/ .ruff_cache/ .mypy_cache/ .pytest_cache/ .coverage src/*.egg-info/

install:
	poetry install --all-extras \
		&& poetry run pip install --no-cache-dir flash-attn --no-build-isolation

install_colab:
	pip install --no-cache-dir . \
		&& pip install --no-cache-dir flash-attn

download_language_models:
	eval $(poetry env activate)	\
		&& python -m nltk.downloader \
		words punkt punkt_tab stopwords wordnet averaged_perceptron_tagger averaged_perceptron_tagger_eng \
		&& python -m spacy download en_core_web_sm

download_hf_models:
	eval $(poetry env activate)	\
		&& huggingface-cli download microsoft/Phi-4-mini-instruct \
		&& huggingface-cli download BAAI/bge-large-en

sync_gdrive:
	sudo rsync -av --delete --progress \
		--exclude=".git/" \
		--exclude=".env" \
		--exclude=".vscode" \
		--exclude=".*_cache/" \
		--exclude=".local/"  \
		--include="data/pdfs_fast" \
		--include="data/pdfs_hi_res" \
		--exclude="data/*" \
		. /mnt/g/O\ meu\ disco/rag-3w-cot/ \
		&& ls -la /mnt/g/O\ meu\ disco/rag-3w-cot

lint:
	poetry run ruff check \
		&& poetry run pyright

lint_fix:
	poetry run ruff check --fix \
		&& poetry run ruff format

.PHONY: clean install install_colab download_language_models download_hf_models sync_gdrive lint lint_fix
