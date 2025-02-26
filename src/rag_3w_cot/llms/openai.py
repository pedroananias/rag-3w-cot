from typing import List

from openai import OpenAI

from .base import BaseLLM


class OpenAIChatGPT(BaseLLM):
    model: str = "openai/gpt-4o-mini"

    @property
    def gpt_model(self):
        return self.model.split("/")[-1]

    def create_pipeline(self) -> OpenAI:
        return OpenAI(api_key=self.settings.open_api_key)

    def call(self, inputs: List[List[dict]], **_) -> List[List[dict]]:
        outputs = []
        for input in inputs:
            output = self.create_pipeline().beta.chat.completions.parse(
                model=self.gpt_model,
                messages=input,  # pyright: ignore
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_chunk_size,
            )
            outputs.append(output)

        return outputs

    def convert_outputs_to_strings(self, outputs: List[List[dict]]) -> List[str]:
        return [output.choices[0].message.content for output in outputs]  # pyright: ignore
