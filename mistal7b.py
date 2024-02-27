import os
PATH =  "D:/classes/cache/huggingface/hub"
# PATH =  "C:/Users/akayl/.cache/huggingface/hub"
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import torch
import time

# tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
# model = AutoModelForCausalLM.from_pretrained("google/gemma-7b", device_map="auto", torch_dtype=torch.float16)
with open("./.env") as file:
    for line in file:
       token = line
model_name = "google/gemma-7b"
config = AutoConfig.from_pretrained(model_name, num_labels=2, output_hidden_states=False, token=token)
model = AutoModelForSequenceClassification.from_config(config)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
tokenizer.pad_token_id = tokenizer.eos_token_id
# model = AutoModelForCausalLM.from_pretrained(model_name,token=toke`n)
# tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)


device = "cuda" # the device to load the model onto

# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")


# messages = [
#     {"role": "user", "content": "What is your favourite condiment?"},
#     {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
#     {"role": "user", "content": "Do you have mayonnaise recipes?"}
# ]

# encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
# model_inputs = encodeds.to(device)


input_text = "Write me a poem about Machine Learning."
model_inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
model.to(device)
start = time.time()
generated_ids = model.generate(**model_inputs)
print("tokens per second: ", (generated_ids.shape[1] - model_inputs["input_ids"].shape[1] )/ (time.time()-start))
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0]) 