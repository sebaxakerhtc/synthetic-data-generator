from distilabel.llms import InferenceEndpointsLLM
from distilabel.steps.tasks import ChatGeneration, Magpie, TextGeneration

from src.distilabel_dataset_generator.pipelines.base import (
    MODEL,
    _get_next_api_key,
)

INFORMATION_SEEKING_PROMPT = (
    "You are an AI assistant designed to provide accurate and concise information on a wide"
    " range of topics. Your purpose is to assist users in finding specific facts,"
    " explanations, or details about various subjects. Provide clear, factual responses and,"
    " when appropriate, offer additional context or related information that might be useful"
    " to the user."
)

REASONING_PROMPT = (
    "You are an AI assistant specialized in logical thinking and problem-solving. Your"
    " purpose is to help users work through complex ideas, analyze situations, and draw"
    " conclusions based on given information. Approach each query with structured thinking,"
    " break down problems into manageable parts, and guide users through the reasoning"
    " process step-by-step."
)

PLANNING_PROMPT = (
    "You are an AI assistant focused on helping users create effective plans and strategies."
    " Your purpose is to assist in organizing thoughts, setting goals, and developing"
    " actionable steps for various projects or activities. Offer structured approaches,"
    " consider potential challenges, and provide tips for efficient execution of plans."
)

EDITING_PROMPT = (
    "You are an AI assistant specialized in editing and improving written content. Your"
    " purpose is to help users refine their writing by offering suggestions for grammar,"
    " style, clarity, and overall structure. Provide constructive feedback, explain your"
    " edits, and offer alternative phrasings when appropriate."
)

CODING_DEBUGGING_PROMPT = (
    "You are an AI assistant designed to help with programming tasks. Your purpose is to"
    " assist users in writing, reviewing, and debugging code across various programming"
    " languages. Provide clear explanations, offer best practices, and help troubleshoot"
    " issues. When appropriate, suggest optimizations or alternative approaches to coding"
    " problems."
)

MATH_SYSTEM_PROMPT = (
    "You are an AI assistant designed to provide helpful, step-by-step guidance on solving"
    " math problems. The user will ask you a wide range of complex mathematical questions."
    " Your purpose is to assist users in understanding mathematical concepts, working through"
    " equations, and arriving at the correct solutions."
)

ROLE_PLAYING_PROMPT = (
    "You are an AI assistant capable of engaging in various role-playing scenarios. Your"
    " purpose is to adopt different personas or characters as requested by the user. Maintain"
    " consistency with the chosen role, respond in character, and help create immersive and"
    " interactive experiences for the user."
)

DATA_ANALYSIS_PROMPT = (
    "You are an AI assistant specialized in data analysis and interpretation. Your purpose is"
    " to help users understand and derive insights from data sets, statistics, and analytical"
    " tasks. Offer clear explanations of data trends, assist with statistical calculations,"
    " and provide guidance on data visualization and interpretation techniques."
)

CREATIVE_WRITING_PROMPT = (
    "You are an AI assistant designed to support creative writing endeavors. Your purpose is"
    " to help users craft engaging stories, poems, and other creative texts. Offer"
    " suggestions for plot development, character creation, dialogue writing, and other"
    " aspects of creative composition. Provide constructive feedback and inspire creativity."
)

ADVICE_SEEKING_PROMPT = (
    "You are an AI assistant focused on providing thoughtful advice and guidance. Your"
    " purpose is to help users navigate various personal or professional issues by offering"
    " balanced perspectives, considering potential outcomes, and suggesting practical"
    " solutions. Encourage users to think critically about their situations while providing"
    " supportive and constructive advice."
)

BRAINSTORMING_PROMPT = (
    "You are an AI assistant specialized in generating ideas and facilitating creative"
    " thinking. Your purpose is to help users explore possibilities, think outside the box,"
    " and develop innovative concepts. Encourage free-flowing thoughts, offer diverse"
    " perspectives, and help users build upon and refine their ideas."
)

PROMPT_CREATION_PROMPT = f"""You are an AI assistant specialized in generating very precise prompts for dataset creation.

Your task is to write a prompt following the instruction of the user. Respond with the prompt and nothing else.

In the generated prompt always finish with this sentence: User questions are direct and concise.

The prompt you write should follow the same style and structure as the following example prompts:

{INFORMATION_SEEKING_PROMPT}

{REASONING_PROMPT}

{PLANNING_PROMPT}

{CODING_DEBUGGING_PROMPT}

{EDITING_PROMPT}

{ROLE_PLAYING_PROMPT}

{DATA_ANALYSIS_PROMPT}

{CREATIVE_WRITING_PROMPT}

{ADVICE_SEEKING_PROMPT}

{BRAINSTORMING_PROMPT}

User dataset description:
"""

DEFAULT_DATASET_DESCRIPTIONS = [
    "rude customer assistant for a phone company",
    "assistant that solves math puzzles using python",
]

_STOP_SEQUENCES = [
    "<|eot_id|>",
    "<|start_header_id|>",
    "assistant",
    " \n\n",
]


def _get_output_mappings(num_turns):
    if num_turns == 1:
        return {"instruction": "prompt", "response": "completion"}
    else:
        return {"conversation": "messages"}


def generate_pipeline_code(system_prompt, num_turns, num_rows):
    input_mappings = _get_output_mappings(num_turns)
    code = f"""
# Requirements: `pip install distilabel[hf-inference-endpoints]`
import os
from distilabel.pipeline import Pipeline
from distilabel.steps import KeepColumns
from distilabel.steps.tasks import MagpieGenerator
from distilabel.llms import InferenceEndpointsLLM

MODEL = "{MODEL}"
SYSTEM_PROMPT = "{system_prompt}"
os.environ["HF_TOKEN"] = "hf_xxx" # https://huggingface.co/settings/tokens/new?ownUserPermissions=repo.content.read&ownUserPermissions=repo.write&globalPermissions=inference.serverless.write&canReadGatedRepos=true&tokenType=fineGrained

with Pipeline(name="sft") as pipeline:
    magpie = MagpieGenerator(
        llm=InferenceEndpointsLLM(
            model_id=MODEL,
            tokenizer_id=MODEL,
            magpie_pre_query_template="llama3",
            generation_kwargs={{
                "temperature": 0.9,
                "do_sample": True,
                "max_new_tokens": 2048,
                "stop_sequences": {_STOP_SEQUENCES}
            }},
            api_key=os.environ["HF_TOKEN"],
        ),
        n_turns={num_turns},
        num_rows={num_rows},
        batch_size=1,
        system_prompt=SYSTEM_PROMPT,
        output_mappings={input_mappings},
    )
    keep_columns = KeepColumns(
        columns={list(input_mappings.values())} + ["model_name"],
    )
    magpie.connect(keep_columns)

if __name__ == "__main__":
    distiset = pipeline.run()
"""
    return code


def get_magpie_generator(num_turns, num_rows, system_prompt, is_sample):
    input_mappings = _get_output_mappings(num_turns)
    output_mappings = input_mappings.copy()
    if num_turns == 1:
        magpie_generator = Magpie(
            llm=InferenceEndpointsLLM(
                model_id=MODEL,
                tokenizer_id=MODEL,
                api_key=_get_next_api_key(),
                magpie_pre_query_template="llama3",
                generation_kwargs={
                    "temperature": 0.9,
                    "do_sample": True,
                    "max_new_tokens": 256 if is_sample else 512,
                    "stop_sequences": _STOP_SEQUENCES,
                },
            ),
            n_turns=num_turns,
            system_prompt=system_prompt,
            output_mappings=output_mappings,
            only_instruction=True,
        )
    else:
        magpie_generator = Magpie(
            llm=InferenceEndpointsLLM(
                model_id=MODEL,
                tokenizer_id=MODEL,
                api_key=_get_next_api_key(),
                magpie_pre_query_template="llama3",
                generation_kwargs={
                    "temperature": 0.9,
                    "do_sample": True,
                    "max_new_tokens": 256 if is_sample else 1024,
                    "stop_sequences": _STOP_SEQUENCES,
                },
            ),
            end_with_user=True,
            n_turns=num_turns,
            system_prompt=system_prompt,
            output_mappings=output_mappings,
        )
    magpie_generator.load()
    return magpie_generator


def get_response_generator(num_turns, system_prompt, is_sample):
    if num_turns == 1:
        response_generator = TextGeneration(
            llm=InferenceEndpointsLLM(
                model_id=MODEL,
                tokenizer_id=MODEL,
                api_key=_get_next_api_key(),
                generation_kwargs={
                    "temperature": 0.8,
                    "max_new_tokens": 256 if is_sample else 1024,
                },
            ),
            system_prompt=system_prompt,
            output_mappings={"generation": "completion"},
            input_mappings={"instruction": "prompt"},
        )
    else:
        response_generator = ChatGeneration(
            llm=InferenceEndpointsLLM(
                model_id=MODEL,
                tokenizer_id=MODEL,
                api_key=_get_next_api_key(),
                generation_kwargs={
                    "temperature": 0.8,
                    "max_new_tokens": 2048,
                },
            ),
            output_mappings={"generation": "completion"},
            input_mappings={"conversation": "messages"},
        )
    response_generator.load()
    return response_generator


def get_prompt_generator():
    prompt_generator = TextGeneration(
        llm=InferenceEndpointsLLM(
            api_key=_get_next_api_key(),
            model_id=MODEL,
            tokenizer_id=MODEL,
            generation_kwargs={
                "temperature": 0.8,
                "max_new_tokens": 2048,
                "do_sample": True,
            },
        ),
        use_system_prompt=True,
    )
    prompt_generator.load()
    return prompt_generator
