from typing import Any, List

import torch
from pydantic import BaseModel, ConfigDict
from transformers import BitsAndBytesConfig, Pipeline, pipeline

from rag_3w_cot.settings import Settings
from rag_3w_cot.utils import force_gpu_cache_release


class BaseLLM(BaseModel):
    settings: Settings

    model: str
    model_torch_dtype: torch.dtype = torch.float16

    device_map: str = "auto"
    task: str = "text-generation"

    model_use_safetensors: bool = True
    model_output_attentions: bool = False
    model_output_hidden_states: bool = False

    has_system_role: bool = True

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    def name(self):
        return self.model.replace("/", "_")

    @property
    def quantization(self):
        if self.settings.device != "cuda":
            return None

        match self.settings.llm_quantization_type:
            case "fp16":
                return None
            case "int8":
                return self.quantization_int8
            case _:
                return self.quantization_int4

    @property
    def quantization_int4(self):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    @property
    def quantization_int8(self):
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=None,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_has_fp16_weight=False,
        )

    @property
    def attn_implementation(self):
        return (
            "flash_attention_2"
            if self.settings.device == "cuda" and self.settings.llm_flash_attention_2
            else "sdpa"
        )

    @property
    def model_kwargs(self):
        return {
            "torch_dtype": self.model_torch_dtype,
            "quantization_config": self.quantization,
            "attn_implementation": self.attn_implementation,
            "use_safetensors": self.model_use_safetensors,
            "output_attentions": self.model_output_attentions,
            "output_hidden_states": self.model_output_hidden_states,
        }

    def __init__(self, **data):
        super().__init__(**data)

    def create_pipeline(self) -> Pipeline | Any:
        _pipeline = pipeline(
            task=self.task,
            model=self.model,
            device_map=self.device_map,
            model_kwargs=self.model_kwargs,
            num_beams=self.settings.llm_num_beams,
            early_stopping=self.settings.llm_early_stopping,
            do_sample=self.settings.llm_do_sample,
            use_fast=self.settings.llm_use_fast,
            padding=self.settings.llm_padding,
            batch_size=self.settings.llm_batch_size,
            temperature=self.settings.llm_temperature,
            top_p=self.settings.llm_pipeline_top_p,
            repetition_penalty=self.settings.llm_repetition_penalty,
            token=self.settings.huggingface_api_key,
        )

        try:
            _pipeline.tokenizer.pad_token_id = _pipeline.model.config.eos_token_id[0]  # pyright: ignore
        except Exception:
            _pipeline.tokenizer.pad_token_id = _pipeline.model.config.eos_token_id  # pyright: ignore

        _pipeline.tokenizer.padding_side = "left"  # pyright: ignore

        return _pipeline

    def call(self, inputs: List[List[dict]]) -> List[List[dict]]:
        if self.settings.force_gpu_cache_release:
            force_gpu_cache_release()

        with torch.no_grad():
            outputs = self.create_pipeline()(
                inputs,
                return_full_text=False,
                max_new_tokens=self.settings.llm_chunk_size,
            )

        if self.settings.force_gpu_cache_release:
            force_gpu_cache_release()

        return outputs  # pyright: ignore

    def convert_outputs_to_strings(self, outputs: List[List[dict]]) -> List[str]:
        return [output[0]["generated_text"] for output in outputs]
