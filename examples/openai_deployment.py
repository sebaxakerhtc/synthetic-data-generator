# pip install synthetic-dataset-generator
import os

from synthetic_dataset_generator import launch

assert os.getenv("HF_TOKEN")  # push the data to huggingface
os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1/"
os.environ["API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["MODEL"] = "gpt-4o"
os.environ["MAGPIE_PRE_QUERY_TEMPLATE"] = None  # chat data not supported with OpenAI

launch()
