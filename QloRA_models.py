import os
if os.getlogin() == "darke":
    PATH =  "D:/classes/cache/huggingface/hub"
    os.environ['HF_HOME'] = PATH
    os.environ['HF_DATASETS_CACHE'] = PATH

import torch


from transformers import BitsAndBytesConfig
from transformers import MistralConfig, MistralForSequenceClassification

from peft import (
    get_peft_model,
    LoraConfig,
    PeftType,
)

def get_config(lora_alpha=16, lora_dropout=0.1, lora_r=8):
    peft_type = PeftType.LORA
    quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

    peft_config = LoraConfig(
        r=lora_r,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="SEQ_CLS",
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        inference_mode=False
    )
    return quantization_config, peft_config

def mistral_QLoRA(args, classes, lora_alpha=16, lora_dropout=0.1, lora_r=8):
    quantization_config, peft_config = get_config(lora_alpha, lora_dropout, lora_r)
    model_name = 'mistralai/Mistral-7B-v0.1'
    config = MistralConfig.from_pretrained(
        model_name,
        num_labels=classes, 
        output_hidden_states=False, 
        quantization_config=quantization_config, 
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        return_dict=True,
        device_map="auto",
        attn_implementation="sdpa" if not args.use_flash_attention_2 else "flash_attention_2")
    model = MistralForSequenceClassification._from_config(config)
    model = get_peft_model(model, peft_config)
    return model