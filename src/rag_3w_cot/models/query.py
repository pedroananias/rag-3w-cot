import json
from pathlib import Path
from typing import List, Optional, Set, Type

from loguru import logger
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator

from rag_3w_cot.dictionaries import BaseTermsDictionary
from rag_3w_cot.models import Document
from rag_3w_cot.utils import normalize_sentence


class Query(BaseModel):
    question_text: str
    kind: str

    _relevant_documents: List[Document] = PrivateAttr(default_factory=list)
    _relevant_files: Set[Path] = PrivateAttr(default_factory=set)
    _dicionaries: List[Type[BaseTermsDictionary]] = PrivateAttr(default_factory=list)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {"question_text": "<string>", "kind": "<number|name|boolean>"},
            ]
        },
    )

    @property
    def question_expanded(self) -> str:
        question = self.question_text
        for dicionary in self._dicionaries:
            question = dicionary.expand(self.question_text)
        return question

    @property
    def question_normalized_expanded(self) -> str:
        return normalize_sentence(self.question_expanded)

    @field_validator("question_text", mode="before")
    @classmethod
    def replace_double_quotes(cls, value: str) -> str:
        return value.replace('"', "'")

    def get_relevant_documents(self) -> List[Document]:
        return self._relevant_documents

    def set_relevant_documents(self, documents: List[Document]):
        self._relevant_documents = documents

    def get_relevant_documents_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(
            [d.model_dump() for d in self.get_relevant_documents()], indent=indent
        )

    def get_relevant_documents_txt(self) -> str:
        lines = []
        for document in self.get_relevant_documents():
            page = document.metadata.get("page_number")
            owner = document.metadata.get("owner")
            lines.append(
                f"**`{owner}` (page {page})**\n```\n{document.page_content}\n```"
            )

        return "\n\n".join(lines)

    def get_relevant_files(self) -> Set[Path]:
        return self._relevant_files

    def set_relevant_files(self, files: Set[Path]):
        self._relevant_files = files

    def set_dicionaries(self, dicionaries: List[Type[BaseTermsDictionary]]):
        self._dicionaries = dicionaries

    def export(self, file: Path):
        file.write_text(self.dump_json())

    def dump_json(self) -> str:
        relevant_files = [str(f) for f in self.get_relevant_files()]
        relevant_documents = [d.model_dump() for d in self.get_relevant_documents()]

        output = {
            "question": self.question_text,
            "question_expanded": self.question_expanded,
            "question_normalized_expanded": self.question_normalized_expanded,
            "relevant_files": relevant_files,
            "relevant_documents": relevant_documents,
        }
        return json.dumps(output, indent=4)

    def debug(self):
        logger.debug(self.dump_json())
