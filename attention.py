import torch
import torch.nn as nn
import torch.nn.functional as F

# class CustomAttention(nn.Module):
#     def __init__(self, config):
#         super(CustomAttention, self).__init__()
#         self.query = nn.Linear(config.hidden_size, config.hidden_size)
#         self.key = nn.Linear(config.hidden_size, config.hidden_size)
#         self.value = nn.Linear(config.hidden_size, config.hidden_size)
        
#         self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
#         self.scale = 1 / (config.hidden_size ** 0.5)
        
#     def forward(self, hidden_states, attention_mask=None, adapted_query_weights=None):
#         if adapted_query_weights is not None:
#             query_layer = torch.matmul(hidden_states, adapted_query_weights)
#         else:
#             query_layer = self.query(hidden_states)
        
#         key_layer = self.key(hidden_states)
#         value_layer = self.value(hidden_states)
        
#         # Calc attention scores
#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) * self.scale
        
#         if attention_mask is not None:
#             # Apply attention mask
#             attention_scores = attention_scores + attention_mask
        
#         # Normalize attention scores to probabilities
#         attention_probs = F.softmax(attention_scores, dim=-1)
#         attention_probs = self.dropout(attention_probs)
        
#         # Weighted sum of the values
#         context_layer = torch.matmul(attention_probs, value_layer)
        
#         return context_layer
