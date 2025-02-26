from .base import BasePrompt


class CotStep1Prompt(BasePrompt):
    file: str = "cot/step_1_cot.md"


class CotStep2Prompt(BasePrompt):
    file: str = "cot/step_2_formatting.md"


class CotStep3Prompt(BasePrompt):
    file: str = "cot/step_3_schema_parsing.md"
