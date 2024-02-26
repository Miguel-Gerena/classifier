import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import attention
class LoRALayer(nn.Module):
    def __init__(self, model_dim, rank, adapter_size=None):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.model_dim = model_dim
        self.adapter_size = adapter_size or model_dim
        
        self.A = nn.Parameter(torch.Tensor(self.adapter_size, self.rank))
        self.B = nn.Parameter(torch.Tensor(self.rank, self.model_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)

    def forward(self, W):
        # W = original weight matrix from the pretrained model
        lora_adaptation = torch.matmul(self.A, self.B)
        return W + lora_adaptation
    

# class LoRATransformerModel(nn.Module):
#     def __init__(self, pretrained_model_name_or_path, rank):
#         super(LoRATransformerModel, self).__init__()
#         self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
#         self.base_model = AutoModel.from_pretrained(pretrained_model_name_or_path, config=self.config)
        
#         self.custom_attention = attention.CustomAttention(self.config)
#         self.lora_adaptations = nn.ModuleList([LoRALayer(self.config.hidden_size, rank) for _ in range(self.config.num_hidden_layers)])
    
#     def forward(self, input_ids, attention_mask=None):
#         # Get embeddings from base model
#         embeddings = self.base_model.embeddings(input_ids)
        
#         for i, layer_module in enumerate(self.base_model.encoder.layer):
#             layer_module_output = layer_module(embeddings, attention_mask=attention_mask)[0]
#             query, key, value = layer_module_output, layer_module_output, layer_module_output 
            
#             adapted_query_weights = self.lora_adaptations[i](layer_module.attention.self.query.weight)
            
#             # Apply custom attention using adapted weights
#             attention_output = self.custom_attention(query, key, value, attention_mask, adapted_query_weights)
#             embeddings = attention_output  
        
        
#         return embeddings

