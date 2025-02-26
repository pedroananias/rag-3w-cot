import importlib
import json
from functools import cached_property
from pathlib import Path
from typing import Any, List, Type

from loguru import logger
from pydantic import BaseModel

from rag_3w_cot.llms import BaseLLM
from rag_3w_cot.models import Answer, Query
from rag_3w_cot.settings import Settings


class BasePipeline(BaseModel):
    settings: Settings
    queries: List[Query]
    output_path: Path

    @cached_property
    def llm(self) -> BaseLLM:
        logger.debug(f"Loading LLM: {self.settings.llm}")

        llm_cls: Type[BaseLLM] = getattr(
            importlib.import_module("rag_3w_cot.llms"),
            self.settings.llm,
        )
        return llm_cls(settings=self.settings)

    def run(self) -> List[Answer]:
        raise NotImplementedError()

    def parse_answers(self, outputs: List[str]) -> List[Answer]:
        answers = []
        for query, output in zip(self.queries, outputs):
            answers.append(Answer.from_string(output, query.question_text, query.kind))

        return answers

    def export_outputs(self, outputs: List[Any], name: str):
        self.output_path.mkdir(parents=True, exist_ok=True)

        # parsed_outputs = []
        # for i, output in enumerate(outputs, start=1):
        #     parsed_outputs.append(f"Output {i} of {len(outputs)} ###################")
        #     parsed_outputs.append(str(output))
        #     parsed_outputs.append("############################################")

        (self.output_path / f"{name}.json").write_text(json.dumps(outputs, indent=4))
