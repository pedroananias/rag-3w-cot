import json
from pathlib import Path

from jinja2 import Template

from rag_3w_cot.models import Answer, Document, Query


class BasePrompt:
    file: str

    _prompt_path: Path = Path(__file__).parent / "markdown"

    @classmethod
    def get_path(cls) -> Path:
        return cls._prompt_path / cls.file

    @classmethod
    def get_content(cls) -> str:
        return cls.get_path().read_text()

    @classmethod
    def get_parsed_content(cls, **additional_placeholders) -> str:
        to_replace = {
            "document_schema": Document.model_json_schema().get("examples", [])[0],
            "query_schema": Query.model_json_schema().get("examples", [])[0],
            "answer_schema": Answer.model_json_schema().get("examples", [])[0],
            **additional_placeholders,
        }

        for placeholder, value in to_replace.items():
            to_replace[placeholder] = (
                json.dumps(value) if not isinstance(value, str) else value
            )

        return Template(cls.get_content()).render(to_replace)

    # @classmethod
    # def parsed(cls, **additional_placeholders) -> str:
    #     content = (cls._prompt_path / cls.file).read_text()
    #     to_replace = {}

    #     query_examples = Query.model_json_schema().get("examples", [])
    #     for i, example in enumerate(query_examples[0:]):
    #         to_replace[f"query_example_{i}"] = example

    #     answer_examples = Answer.model_json_schema().get("examples", [])
    #     for i, example in enumerate(answer_examples[0:]):
    #         to_replace[f"answer_example_{i}"] = example

    #     to_replace["document_schema"] = Document.model_json_schema().get(
    #         "examples", []
    #     )[0]
    #     to_replace["query_schema"] = Query.model_json_schema().get("examples", [])[0]
    #     to_replace["answer_schema"] = Answer.model_json_schema().get("examples", [])[0]

    #     to_replace.update(additional_placeholders)

    #     for placeholder, value in to_replace.items():
    #         content = content.replace(
    #             "{" + placeholder + "}",
    #             json.dumps(value) if not isinstance(value, str) else value,
    #         )

    #     return content
