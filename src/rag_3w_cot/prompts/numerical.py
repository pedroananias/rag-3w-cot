from .base import BasePrompt


class NumericalNormalizationSystemPrompt(BasePrompt):
    file: str = "processing_numerical_normalization_system_prompt.md"


class NumericalNormalizationUserPrompt(BasePrompt):
    file: str = "processing_numerical_normalization_user_prompt.md"
