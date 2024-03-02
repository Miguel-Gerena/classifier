import os
PATH =  "D:/classes/cache/huggingface/hub"
# PATH =  "C:/Users/akayl/.cache/huggingface/hub"
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import torch
import time
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import lora

device = "cuda" # the device to load the model onto

# model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")



class CustomizedMistralModel(nn.Module):
    def __init__(self, model_name, rank, num_labels, dropout_prob=0.1):
        super(CustomizedMistralModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name, config=self.config)
        
        # LoRA layers for each attention layer
        self.lora_layers = nn.ModuleList([lora.LoRALayer(self.config.hidden_size, rank) for _ in range(self.config.num_hidden_layers)])
        
        # Adding a classifier on top of the base model
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_prob)
        
    def apply_lora(self):
        # Iterate through each MistralDecoderLayer in the base_model's layers
        for layer, lora_layer in zip(self.base_model.layers, self.lora_layers):
            # Apply LoRA to adapt the query weights (q_proj) in the self-attention mechanism
            with torch.no_grad():
                original_q_proj_weights = layer.self_attn.q_proj.weight.data

                # print("OG WEIGHTS:", original_q_proj_weights)

                adapted_q_proj_weights = lora_layer(original_q_proj_weights)
                layer.self_attn.q_proj.weight.data = adapted_q_proj_weights
                # print("ADAPTED WEIGHTS", adapted_q_proj_weights)
                # exit()


    def forward(self, input_ids, attention_mask=None, labels=None):

        
        self.apply_lora()

        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return SimpleOutput(logits=logits) 



class SimpleOutput:
    def __init__(self, logits):
        self.logits = logits










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