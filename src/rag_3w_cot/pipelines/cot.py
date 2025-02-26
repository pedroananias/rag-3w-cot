from typing import List

from loguru import logger

from rag_3w_cot.models import Answer
from rag_3w_cot.prompts import CotStep1Prompt, CotStep2Prompt, CotStep3Prompt

from .base import BasePipeline


class CotPipeline(BasePipeline):
    def run(self) -> List[Answer]:
        logger.warning("Running CoT Pipeline")

        outputs = self.step_1_cot()
        self.export_outputs(outputs, "step_1_cot")

        outputs = self.step_2_formatting(outputs)
        self.export_outputs(outputs, "step_2_formatting")

        outputs = self.step_3_schema_parsing(outputs)
        self.export_outputs(outputs, "step_3_schema_parsing")

        return self.parse_answers(outputs)

    def step_1_cot(self) -> List[str]:
        logger.warning("Running Step 1: COT")

        system_prompt = CotStep1Prompt.get_parsed_content()

        inputs = []
        for query in self.queries:
            query_inputs = [{"role": "system", "content": system_prompt}]

            for document in query.get_relevant_documents():
                query_inputs.append(
                    {"role": "user", "content": document.model_dump_json()}
                )

            query_inputs.append({"role": "user", "content": query.model_dump_json()})

            inputs.append(query_inputs)

        llm_outputs = self.llm.call(inputs)
        outputs = self.llm.convert_outputs_to_strings(llm_outputs)

        return outputs

    def step_2_formatting(self, previous_outputs: List[str]) -> List[str]:
        logger.warning("Running Step 2: Formatting")

        system_prompt = CotStep2Prompt.get_parsed_content()

        inputs = []
        for previous_output in previous_outputs:
            input = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": previous_output},
            ]
            inputs.append(input)

        llm_outputs = self.llm.call(inputs)
        outputs = self.llm.convert_outputs_to_strings(llm_outputs)

        return outputs

    def step_3_schema_parsing(self, previous_outputs: List[str]) -> List[str]:
        logger.warning("Running Step 3: Schema Parsing")

        inputs = []
        for previous_output, query in zip(previous_outputs, self.queries):
            system_prompt = CotStep3Prompt.get_parsed_content(
                query=query.model_dump_json()
            )

            input = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": previous_output},
            ]
            inputs.append(input)

        llm_outputs = self.llm.call(inputs)
        outputs = self.llm.convert_outputs_to_strings(llm_outputs)

        return outputs
