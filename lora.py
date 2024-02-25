import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

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
    

class LoRATransformerModel(nn.Module):
    def __init__(self, pretrained_model_name_or_path, rank):
        super(LoRATransformerModel, self).__init__()
        self.config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        self.base_model = AutoModel.from_pretrained(pretrained_model_name_or_path, config=self.config)
        
        model_dim = self.config.hidden_size
        
        # Create a LoRA layer for each attention layer's query weight
        self.lora_adaptations = nn.ModuleList([LoRALayer(model_dim, rank) for _ in range(self.config.num_hidden_layers)])
        
        # Freeze the pretrained weights if only training the LoRA parameters
        # for param in self.base_model.parameters():
        #     param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)

        for i, layer in enumerate(self.base_model.encoder.layer):
            # OG weights
            original_query_weights = layer.attention.self.query.weight
            
            # Apply LoRA adaptation
            adapted_query_weights = self.lora_adaptations[i](original_query_weights)
            
            #TODO: CREATE A CUSTOM ATTENTION MODULE- thats able to accept new query,key, value weights as inputs
        return outputs

