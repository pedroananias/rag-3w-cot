import gc
import re
import string
from datetime import datetime
from functools import lru_cache
from typing import List

import numpy as np
import spacy
import torch
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STOP_WORDS = set(stopwords.words("english"))
PUNCTUATION = set(string.punctuation)
FORBIDDEN_TAGS = [
    "CC",
    "DT",
    "EX",
    "IN",
    "LS",
    "MD",
    "PDT",
    "POS",
    "PRP",
    "PRP$",
    "RB",
    "RBR",
    "RBS",
    "RP",
    "SYM",
    "TO",
    "UH",
    # "VB",
    "WDT",
    "WP",
    "WP$",
    "WRB",
]


def normalize_sentence(sentence: str) -> str:
    tokens = pos_tag(word_tokenize(sentence))

    important_words = [
        str(word).lower().strip("'")
        for word, tag in tokens
        if word.lower() not in STOP_WORDS
        and word not in PUNCTUATION
        and tag not in FORBIDDEN_TAGS
    ]

    return " ".join(important_words)


def parse_json_object_in_string(intput: str) -> str:
    match = re.search(r"\{.*\}", intput, re.DOTALL)
    if match:
        return match.group()

    return intput


def get_cosine_similarity(text1: str, text2: str, stop_words: bool = True) -> float:
    vectorizer = TfidfVectorizer(stop_words="english" if stop_words else None)
    matrix = vectorizer.fit_transform([str(text1).lower()])
    query_vector = vectorizer.transform([str(text2).lower()])
    return float(np.mean(cosine_similarity(query_vector, matrix)))


def get_vector_cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    return float(
        np.mean(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    )


@lru_cache
def spacey_language_model():
    return spacy.load("en_core_web_sm")


def extract_most_common_year(
    documents: List[dict],
    content_key: str = "page_content",
    year_start: int = datetime.now().year - 10,
    year_end: int = datetime.now().year - 1,
) -> int:
    contents = [str(item.get(content_key, "")) for item in documents]
    possible_header_footnotes = [
        content for content in contents if content and len(content) <= 250
    ]
    merged_content = " ".join(possible_header_footnotes)

    doc = spacey_language_model()(merged_content)

    found_years = [
        int(ent.text)
        for ent in doc.ents
        if ent.label_ == "DATE"
        and ent.text.isdigit()
        and year_start <= int(ent.text) <= year_end
    ]

    return round(sum(found_years) / len(found_years)) if found_years else -1


def force_gpu_cache_release():
    gc.collect()
    torch.cuda.empty_cache()
