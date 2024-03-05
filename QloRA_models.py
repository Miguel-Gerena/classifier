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
    prepare_model_for_kbit_training,
    PeftType,TaskType
)


def get_config(lora_alpha=16, lora_dropout=0.1, lora_r=64):
    peft_type = PeftType.LORA
    peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    bias="none",
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    target_modules=[
        "q_proj",
        "v_proj"
    ],
)

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
    return peft_config, bnb_config

def mistral_QLoRA(args, classes, tokenizer, lora_alpha=16, lora_dropout=0.1, lora_r=64):
    peft_config, bnb_config = get_config(lora_alpha, lora_dropout, lora_r)
    model_name = 'mistralai/Mistral-7B-v0.1'
    model = MistralForSequenceClassification.from_pretrained(
        model_name,
        num_labels=classes,
        quantization_config=bnb_config,
        )

    #Setting the Pretraining_tp to 1 ensures we are using the Linear Layers to the max computation possible

    #Ensuring the model is aware about the pad token ID
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.config.pretraining_tp = 1 #For Us this would be 7B
    model.config.pad_token_id = tokenizer.pad_token_id
    return model