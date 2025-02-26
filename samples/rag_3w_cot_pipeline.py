import json
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from loguru import logger

from rag_3w_cot.evaluations import (
    BaseEvaluation,
    BERTScoreEvaluation,
    EmbeddingCosineSimilarityEvaluation,
    ExactMatchEvaluation,
    RougeScoreEvaluation,
)
from rag_3w_cot.models import Answer, Query
from rag_3w_cot.pipelines import CotPipeline
from rag_3w_cot.processors import DocumentProcessor, QueryProcessor
from rag_3w_cot.settings import Settings

load_dotenv()
warnings.filterwarnings("ignore")


def setup_output_path(data_path: Path) -> Path:
    output_path = data_path / "output" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def load_queries(data_path: Path, queries_json: str = "questions.json") -> List[Query]:
    queries_path = data_path / queries_json
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file {queries_path} not found")

    queries: List[Query] = []
    with open(data_path / queries_json, "r") as f:
        queries_list = json.load(f)
        for question in queries_list:
            queries.append(
                Query(
                    question_text=str(question["text"]),
                    kind=str(question["kind"]),  # type: ignore
                )
            )

    return queries


def load_true_answers(
    data_path: Path, true_answers_json: str = "true_answers.json"
) -> List[Answer]:
    true_answers_path = data_path / true_answers_json
    if not true_answers_path.exists():
        return []

    true_answers: List[Answer] = []
    with open(data_path / true_answers_json, "r") as f:
        true_answers_list = json.load(f)
        for true_answer in true_answers_list:
            true_answers.append(
                Answer(
                    question_text=true_answer["question_text"],
                    value=true_answer["value"],
                    kind=true_answer["kind"],
                )
            )

    return true_answers


def run_pipeline(
    data_path: Path,
    settings: Settings,
    queries_json: str = "questions.json",
    metadata_json: str = "subset.json",
    true_answers_json: str = "true_answers.json",
):
    ###### Data

    output_path = setup_output_path(data_path)
    queries = load_queries(data_path, queries_json)
    true_answers = load_true_answers(data_path, true_answers_json)

    ###### Settings

    settings.debug()
    settings.export(output_path / "settings.json")

    ###### Processors
    
    start_time = time.time()

    document_processor = DocumentProcessor(
        settings=settings,
        data_path=data_path,
        metadata_file=data_path / metadata_json,
    )
    document_processor.process()

    query_processor = QueryProcessor(
        settings=settings,
        data_path=data_path,
        metadata_file=data_path / metadata_json,
    )
    queries: List[Query] = query_processor.process(queries)
    for i, query in enumerate(queries, start=1):
        query.export(output_path / f"query_{i}.json")

    ###### Pipeline

    pipeline = CotPipeline(settings=settings, queries=queries, output_path=output_path)
    answers = pipeline.run()

    ###### Outputs

    end_time = time.time()
    latency = end_time - start_time

    answers_as_dict = [answer.model_dump(by_alias=True) for answer in answers]
    answers_json_path = output_path / "answers.json"
    answers_json_path.write_text(json.dumps(answers_as_dict, default=str, indent=4))
    logger.debug(f"Answers saved to {answers_json_path}")

    latency_json_path = output_path / "latency.txt"
    latency_json_path.write_text(json.dumps(latency, indent=4))

    ###### Evaluations

    if not true_answers:
        return

    evaluations = [
        EmbeddingCosineSimilarityEvaluation,
        ExactMatchEvaluation,
        BERTScoreEvaluation,
        RougeScoreEvaluation,
    ]

    scores = {}
    for evaluation in evaluations:
        try:
            target: BaseEvaluation = evaluation(
                settings=settings,
                answers=answers,
                true_answers=true_answers,
            )
            scores[evaluation.__name__] = target.get_score()
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Error calling evaluation {evaluation.__name__}: {e} -> {tb}")
            scores[evaluation.__name__] = -1

    logger.success(f"Scores: {json.dumps(scores, indent=4)}")

    scores_json_path = output_path / "scores.json"
    scores_json_path.write_text(json.dumps(scores, indent=4))


if __name__ == "__main__":
    data_path = Path("samples/data")
    # data_path = Path("data/pdfs_hi_res")

    settings = Settings(
        llm="Qwen257B",
        llm_batch_size=2,
        llm_quantization_type="int4",
        processing_max_concurrent_tasks=4,
        unstructured_strategy="hi_res",
    )  # type: ignore

    run_pipeline(data_path, settings)
