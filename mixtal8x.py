import transformers
from transformers import AutoTokenizer
import torch

model = "mistralai/Mixtral-8x7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model, device_map="auto")
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True},
)

messages = [{"role": "user", "content": "Explain what a Mixture of Experts is in less than 100 words."}]
prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipeline(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])

#quant better performance
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
# tokenizer = AutoTokenizer.from_pretrained(model_id)

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16
# )
# model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)

# prompt = "[INST] Explain what a Mixture of Experts is in less than 100 words. [/INST]"
# inputs = tokenizer(prompt, return_tensors="pt").to(0)

# output = model.generate(**inputs, max_new_tokens=50)
# print(tokenizer.decode(output[0], skip_special_tokens=True))



# the bloke gptq
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# model_id = "TheBloke/Mixtral-8x7B-v0.1-GPTQ"
# tokenizer = AutoTokenizer.from_pretrained(model_id)

# model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# prompt = "[INST] Explain what a Mixture of Experts is in less than 100 words. [/INST]"
# inputs = tokenizer(prompt, return_tensors="pt").to(0)

# output = model.generate(**inputs, max_new_tokens=50)
# print(tokenizer.decode(output[0], skip_special_tokens=True))

