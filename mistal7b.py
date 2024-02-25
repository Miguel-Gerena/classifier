import os
# PATH =  "D:/classes/cache/huggingface/hub"
PATH =  "C:/Users/akayl/.cache/huggingface/hub"
os.environ['TRANSFORMERS_CACHE'] = PATH
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH
from transformers import AutoModelForCausalLM, AutoTokenizer
import time


device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")



# messages = [
#     {"role": "user", "content": "What is your favourite condiment?"},
#     {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
#     {"role": "user", "content": "Do you have mayonnaise recipes?"}
# ]

# encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

# model_inputs = encodeds.to(device)
# model.to(device)
# start = time.time()
# generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
# print("tokens per second: ", (generated_ids.shape[1] - encodeds.shape[1] )/ (time.time()-start))
# decoded = tokenizer.batch_decode(generated_ids)
# print(decoded[0]) 