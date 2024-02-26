import os
# PATH =  "D:/classes/cache/huggingface/hub"
PATH =  "C:/Users/akayl/.cache/huggingface/hub"
os.environ['TRANSFORMERS_CACHE'] = PATH
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import lora

# device = "cuda" # the device to load the model onto

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
        
        # Optionally, apply dropout for regularization
        self.dropout = nn.Dropout(dropout_prob)
        
    def apply_lora(self):
        # Iterate through each MistralDecoderLayer in the base_model's layers
        for layer, lora_layer in zip(self.base_model.layers, self.lora_layers):
            # Apply LoRA to adapt the query weights (q_proj) in the self-attention mechanism
            with torch.no_grad():
                original_q_proj_weights = layer.self_attn.q_proj.weight.data
                adapted_q_proj_weights = lora_layer(original_q_proj_weights)
                layer.self_attn.q_proj.weight.data = adapted_q_proj_weights


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