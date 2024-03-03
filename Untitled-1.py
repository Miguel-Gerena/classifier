# %%
import argparse
import os
if os.getlogin() == "darke":
    PATH =  "D:/classes/cache/huggingface/hub"
    os.environ['HF_HOME'] = PATH
    os.environ['HF_DATASETS_CACHE'] = PATH

import torch

from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_model,
    LoraConfig,
    PeftType,
)

from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup,set_seed, BitsAndBytesConfig
from transformers import MistralConfig, MistralForSequenceClassification

from tqdm import tqdm

from data_handling import map_decision_to_string, create_model_and_tokenizer, dataset_statistics, measure_accuracy, create_dataset


device = "cuda"
model_id = "mistralai/Mistral-7B-v0.1"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right")
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

peft_config = LoraConfig(
    r=script_args.lora_r,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="SEQ_CLS",
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout,
    inference_mode=False
)


# %%
peft_type = PeftType.LORA


dataset_dict = load_dataset(args.dataset_load_path , 
    name=args.dataset_name,
    ipc_label=args.ipc_label,
    cpc_label= args.cpc_label,
    train_filing_start_date=args.train_filing_start_date, 
    train_filing_end_date=args.train_filing_end_date,
    val_filing_start_date=args.val_filing_start_date, 
    val_filing_end_date=args.val_filing_end_date,
    val_set_balancer = args.val_set_balancer,
    uniform_split = args.uniform_split,
    )

for name in ['train', 'validation']:
    dataset_dict[name] = dataset_dict[name].map(map_decision_to_string, num_proc=args.num_proc)
    # Remove the pending and CONT-patent applications
    dataset_dict[name] = dataset_dict[name].filter(lambda e: e['output'] <= 1)



        # Load the dataset
data_loaders = create_dataset(
    args = args, 
    dataset_dict = dataset_dict, 
    tokenizer = tokenizer, 
    section = args.section,
    return_data_loader=False,
    collate_fn=collate_fn
    )
del dataset_dict

# Instantiate dataloaders.
train_dataloader = data_loaders[0]
eval_dataloader = data_loaders[1]

# %%

CLASSES = 2
config = MistralConfig.from_pretrained(
    model_id,
    num_labels=CLASSES, 
    output_hidden_states=False, 
    quantization_config=quantization_config, 
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    return_dict=True,
    attn_implementation="sdpa" if not script_args.use_flash_attention_2 else "flash_attention_2")

model = MistralForSequenceClassification._from_config(config)
model = get_peft_model(model, peft_config)
print(model.print_trainable_parameters())
print(model)

# %%
optimizer = AdamW(params=model.parameters(), lr=lr)

# Instantiate scheduler
lr_scheduler = get_constant_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
    num_training_steps=(len(train_dataloader) * num_epochs),
)

# no_decay = ["bias", "LayerNorm.weight"]
# params_decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
# params_nodecay = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
# optim_groups = [
# {"params": params_decay, "weight_decay": config.weight_decay},
# {"params": params_nodecay, "weight_decay": 0.0},
# ]
# model.zero_grad()
# loss.backward()
# torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
# optimizer.step()
# # %%

model.to(device)
for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    print(f"epoch {epoch}:", eval_metric)

