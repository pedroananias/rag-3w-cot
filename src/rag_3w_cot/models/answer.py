import json
import re
from typing import List

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class AnswerReference(BaseModel):
    pdf_sha1: str
    page_index: int

    @field_validator("page_index", mode="before")
    @classmethod
    def cast_page_index(cls, value):
        if isinstance(value, int):
            return value

        try:
            int_value = int(value)
            if str(int_value) == str(value):
                return int_value
        except (ValueError, TypeError):
            pass

        return -1


class Answer(BaseModel):
    question_text: str
    kind: str
    value: bool | int | float | str
    references: List[AnswerReference] = Field(default_factory=list)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "question_text": "<original question>",
                    "kind": "<original kind>",
                    "value": "<your answer>",
                    "references": [
                        {
                            "pdf_sha1": "<file sha1>",
                            "page_index": "<page number>",
                        }
                    ],
                },
            ]
        },
    )

    @field_validator("question_text", mode="before")
    @classmethod
    def replace_double_quotes(cls, value: str) -> str:
        return value.replace('"', "'")

    @field_validator("value", mode="before")
    @classmethod
    def cast_answer(cls, value):
        if isinstance(value, bool):
            return value

        if isinstance(value, str) and value.lower() in ["true", "false"]:
            return value.lower() == "true"

        if isinstance(value, str) and value.lower() in ["yes", "no"]:
            return value.lower() == "yes"

        try:
            int_value = int(value)
            if str(int_value) == str(value):
                return int_value
        except (ValueError, TypeError):
            pass

        try:
            return float(value)
        except (ValueError, TypeError):
            pass

        return str(value)

    @staticmethod
    def from_string(input_str: str, question: str, kind: str) -> "Answer":
        try:
            return Answer.model_validate_json(input_str)
        except ValidationError as e:
            logger.warning(
                f'{question}: error ({e}) validating input "{input_str}". Falling back to JSON parsing...'
            )
            try:
                answer_dict = json.loads(input_str)
                return Answer(
                    question_text=question,
                    kind=kind,
                    value=answer_dict["value"],
                    references=answer_dict["references"],
                )
            except Exception:
                logger.error(f"{question}: error parsing JSON! Trying with regex...")
                try:
                    match = re.search(r"```json\n(.*?)\n```", input_str, re.DOTALL)
                    if match:
                        input_str = match.group(1)
                        answer_dict = json.loads(input_str)
                        return Answer(
                            question_text=question,
                            kind=kind,
                            value=answer_dict["value"],
                            references=answer_dict["references"],
                        )
                except Exception:
                    logger.error(
                        f"{question}: error parsing with regex! Skipping answer..."
                    )

        return Answer(question_text=question, kind=kind, value="N/A", references=[])
