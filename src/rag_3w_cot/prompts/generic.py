from .base import BasePrompt


class GenericSystemPrompt(BasePrompt):
    file: str = "generic_system_prompt.md"


class GenericShortSystemPrompt(BasePrompt):
    file: str = "generic_short_system_prompt.md"


class GenericUserPrompt(BasePrompt):
    file: str = "generic_user_prompt.md"
