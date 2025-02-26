from .base import BasePrompt


class FinancialSystemPrompt(BasePrompt):
    file: str = "financial_system_prompt.md"


class FinancialUserPrompt(BasePrompt):
    file: str = "financial_user_prompt.md"
