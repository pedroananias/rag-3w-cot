from typing import Dict

from pydantic import BaseModel, ConfigDict


class Document(BaseModel):
    id: str
    metadata: Dict[str, int | float | str]
    page_content: str

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "id": "<uuid>",
                    "metadata": {
                        "pdf_sha1": "<sha1>",
                        "page_index": "<int>",
                        "owner": "<owner>",
                        "year": "<int>",
                        "content_type": "<text|html|markdown>",
                        "score": "<float>",
                    },
                    "page_content": "<content>",
                }
            ]
        },
    )
