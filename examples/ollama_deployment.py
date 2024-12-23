# pip install synthetic-dataset-generator
# ollama serve
# ollama run llama3.1:8b-instruct-q8_0
import os

from synthetic_dataset_generator import launch

assert os.getenv("HF_TOKEN")  # push the data to huggingface
os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:11434/"  # ollama base url
os.environ["MODEL"] = "llama3.1:8b-instruct-q8_0"  # model id
os.environ["TOKENIZER_ID"] = "meta-llama/Llama-3.1-8B-Instruct"  # tokenizer id
os.environ["MAGPIE_PRE_QUERY_TEMPLATE"] = "llama3"  # magpie template

launch()
