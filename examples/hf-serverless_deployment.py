# pip install synthetic-dataset-generator
import os

from synthetic_dataset_generator import launch

assert os.getenv("HF_TOKEN")  # push the data to huggingface
os.environ["MODEL"] = "meta-llama/Llama-3.1-8B-Instruct"  # use instruct model
os.environ["MAGPIE_PRE_QUERY_TEMPLATE"] = "llama3"  # use the template for the model

launch()
