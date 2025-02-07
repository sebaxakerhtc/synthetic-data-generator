import math
import random

from distilabel.models import ClientvLLM, InferenceEndpointsLLM, OllamaLLM, OpenAILLM
from distilabel.steps.tasks import TextGeneration

from synthetic_dataset_generator.constants import (
    API_KEYS,
    DEFAULT_BATCH_SIZE,
    HUGGINGFACE_BASE_URL,
    HUGGINGFACE_BASE_URL_COMPLETION,
    MODEL,
    MODEL_COMPLETION,
    OLLAMA_BASE_URL,
    OLLAMA_BASE_URL_COMPLETION,
    OPENAI_BASE_URL,
    OPENAI_BASE_URL_COMPLETION,
    TOKENIZER_ID,
    TOKENIZER_ID_COMPLETION,
    VLLM_BASE_URL,
    VLLM_BASE_URL_COMPLETION,
)

TOKEN_INDEX = 0


def _get_next_api_key():
    global TOKEN_INDEX
    api_key = API_KEYS[TOKEN_INDEX % len(API_KEYS)]
    TOKEN_INDEX += 1
    return api_key


def _get_prompt_rewriter():
    generation_kwargs = {
        "temperature": 1,
    }
    system_prompt = "You are a prompt rewriter. You are given a prompt and you need to rewrite it keeping the same structure but highlighting different aspects of the original without adding anything new."
    prompt_rewriter = TextGeneration(
        llm=_get_llm(generation_kwargs=generation_kwargs),
        system_prompt=system_prompt,
        use_system_prompt=True,
    )
    prompt_rewriter.load()
    return prompt_rewriter


def get_rewritten_prompts(prompt: str, num_rows: int):
    prompt_rewriter = _get_prompt_rewriter()
    # create prompt rewrites
    inputs = [
        {"instruction": f"Original prompt: {prompt} \nRewritten prompt: "}
        for i in range(math.floor(num_rows / 100))
    ]
    n_processed = 0
    prompt_rewrites = [prompt]
    while n_processed < num_rows:
        batch = list(
            prompt_rewriter.process(
                inputs=inputs[n_processed : n_processed + DEFAULT_BATCH_SIZE]
            )
        )
        prompt_rewrites += [entry["generation"] for entry in batch[0]]
        n_processed += DEFAULT_BATCH_SIZE
        random.seed(a=random.randint(0, 2**32 - 1))
    return prompt_rewrites


def _get_llm_class() -> str:
    if OPENAI_BASE_URL:
        return "OpenAILLM"
    elif OLLAMA_BASE_URL:
        return "OllamaLLM"
    elif HUGGINGFACE_BASE_URL:
        return "InferenceEndpointsLLM"
    elif VLLM_BASE_URL:
        return "ClientvLLM"
    else:
        return "InferenceEndpointsLLM"


def _get_llm(
    structured_output: dict = None,
    use_magpie_template: str = False,
    is_completion: bool = False,
    **kwargs,
):
    model = MODEL_COMPLETION if is_completion else MODEL
    tokenizer_id = TOKENIZER_ID_COMPLETION if is_completion else TOKENIZER_ID or model
    base_urls = {
        "openai": OPENAI_BASE_URL_COMPLETION if is_completion else OPENAI_BASE_URL,
        "ollama": OLLAMA_BASE_URL_COMPLETION if is_completion else OLLAMA_BASE_URL,
        "huggingface": HUGGINGFACE_BASE_URL_COMPLETION if is_completion else HUGGINGFACE_BASE_URL,
        "vllm": VLLM_BASE_URL_COMPLETION if is_completion else VLLM_BASE_URL,
    }

    if base_urls["openai"]:
        llm = OpenAILLM(
            model=model,
            base_url=base_urls["openai"],
            api_key=_get_next_api_key(),
            structured_output=structured_output,
            **kwargs,
        )
        if "generation_kwargs" in kwargs:
            if "stop_sequences" in kwargs["generation_kwargs"]:
                kwargs["generation_kwargs"]["stop"] = kwargs["generation_kwargs"][
                    "stop_sequences"
                ]
                del kwargs["generation_kwargs"]["stop_sequences"]
            if "do_sample" in kwargs["generation_kwargs"]:
                del kwargs["generation_kwargs"]["do_sample"]
    elif base_urls["ollama"]:
        if "generation_kwargs" in kwargs:
            if "max_new_tokens" in kwargs["generation_kwargs"]:
                kwargs["generation_kwargs"]["num_predict"] = kwargs[
                    "generation_kwargs"
                ]["max_new_tokens"]
                del kwargs["generation_kwargs"]["max_new_tokens"]
            if "stop_sequences" in kwargs["generation_kwargs"]:
                kwargs["generation_kwargs"]["stop"] = kwargs["generation_kwargs"][
                    "stop_sequences"
                ]
                del kwargs["generation_kwargs"]["stop_sequences"]
            if "do_sample" in kwargs["generation_kwargs"]:
                del kwargs["generation_kwargs"]["do_sample"]
            options = kwargs["generation_kwargs"]
            del kwargs["generation_kwargs"]
            kwargs["generation_kwargs"] = {}
            kwargs["generation_kwargs"]["options"] = options
        llm = OllamaLLM(
            model=model,
            host=base_urls["ollama"],
            tokenizer_id=tokenizer_id,
            use_magpie_template=use_magpie_template,
            structured_output=structured_output,
            **kwargs,
        )
    elif base_urls["huggingface"]:
        kwargs["generation_kwargs"]["do_sample"] = True
        llm = InferenceEndpointsLLM(
            api_key=_get_next_api_key(),
            base_url=base_urls["huggingface"],
            tokenizer_id=tokenizer_id,
            use_magpie_template=use_magpie_template,
            structured_output=structured_output,
            **kwargs,
        )
    elif base_urls["vllm"]:
        if "generation_kwargs" in kwargs:
            if "do_sample" in kwargs["generation_kwargs"]:
                del kwargs["generation_kwargs"]["do_sample"]
        llm = ClientvLLM(
            base_url=base_urls["vllm"],
            model=model,
            tokenizer=tokenizer_id,
            api_key=_get_next_api_key(),
            use_magpie_template=use_magpie_template,
            structured_output=structured_output,
            **kwargs,
        )
    else:
        llm = InferenceEndpointsLLM(
            api_key=_get_next_api_key(),
            tokenizer_id=tokenizer_id,
            model_id=model,
            use_magpie_template=use_magpie_template,
            structured_output=structured_output,
            **kwargs,
        )

    return llm


try:
    llm = _get_llm()
    llm.load()
    llm.generate([[{"content": "Hello, world!", "role": "user"}]])
except Exception as e:
    raise Exception(f"Error loading {llm.__class__.__name__}: {e}")
