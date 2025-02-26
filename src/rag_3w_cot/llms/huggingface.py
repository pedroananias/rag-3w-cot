from .base import BaseLLM


class DeepSeekR1Llama8B(BaseLLM):
    model: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"


class MicrosoftPhi4(BaseLLM):
    model: str = "microsoft/phi-4"


class MicrosoftPhi4Mini(BaseLLM):
    model: str = "microsoft/Phi-4-mini-instruct"


class Qwen257B(BaseLLM):
    model: str = "Qwen/Qwen2.5-7B-Instruct-1M"


class Qwen2514B(BaseLLM):
    model: str = "Qwen/Qwen2.5-14B-Instruct-1M"
