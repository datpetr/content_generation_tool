import transformers
import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

login(token='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')

tokenizer = AutoTokenizer.from_pretrained(
"meta-llama/Meta-Llama-3.1-8B",
cache_dir="/kaggle/working/"
)

model = AutoModelForCausalLM.from_pretrained(
"meta-llama/Meta-Llama-3.1-8B",
cache_dir="/kaggle/working/",
device_map="auto",
)

model_id = "meta-llama/Meta-Llama-3.1-8B"

pipeline = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)

pipeline("Hey how are you doing today?")
