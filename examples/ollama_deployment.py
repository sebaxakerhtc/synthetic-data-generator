# pip install synthetic-dataset-generator
# ollama serve
# ollama run llama3.1:8b-instruct-q8_0
import os

from synthetic_dataset_generator import launch

assert os.getenv("HF_TOKEN")  # push the data to huggingface
os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:11434/"
os.environ["MODEL"] = "llama3.1:8b-instruct-q8_0"
os.environ["TOKENIZER_ID"] = "meta-llama/Llama-3.1-8B-Instruct"
os.environ["MAGPIE_PRE_QUERY_TEMPLATE"] = "llama3"
os.environ["MAX_NUM_ROWS"] = "10000"
os.environ["DEFAULT_BATCH_SIZE"] = "5"
os.environ["MAX_NUM_TOKENS"] = "2048"

launch()
