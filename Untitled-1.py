import os
if os.getlogin() == "darke":
    PATH =  "D:/classes/cache/huggingface/hub"
    os.environ['HF_HOME'] = PATH
    os.environ['HF_DATASETS_CACHE'] = PATH

import torch
from torch.optim import AdamW
from dataclasses import dataclass, field
from typing import Optional
from peft import (
    get_peft_model,
    LoraConfig,
    PeftType,
)

import json
from tqdm import tqdm
import argparse

from transformers import AutoTokenizer, HfArgumentParser, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, is_bitsandbytes_available
from transformers import AutoModelForSequenceClassification, AutoConfig
from transformers import MistralForSequenceClassification, MistralConfig, get_constant_schedule_with_warmup
from data_handling import map_decision_to_string, create_model_and_tokenizer, dataset_statistics, measure_accuracy, create_dataset


from datasets import load_dataset
from trl import SFTTrainer

@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.01)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=8)

    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables bf16 training."},
    )

    use_flash_attention_2: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash Attention 2."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )

    warmup_ratio: float = field(default=0.03, metadata={"help": "Fraction of steps to do a warmup for"})


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    parser = argparse.ArgumentParser()
        
    # Dataset
    parser.add_argument('--dataset_name', default='sample', type=str, help='Patent data directory.')
    parser.add_argument('--dataset_load_path', default='./hupd.py', type=str, help='Patent data main data load path (viz., ../patents.py).')
    parser.add_argument('--cpc_label', type=str, default=None, help='CPC label for filtering the data.')
    parser.add_argument('--ipc_label', type=str, default=None, help='IPC label for filtering the data.')
    parser.add_argument('--section', type=str, default='claims', help='Patent application section of interest.')
    parser.add_argument('--train_filing_start_date', type=str, default='', help='Start date for filtering the training data.')
    parser.add_argument('--train_filing_end_date', type=str, default='', help='End date for filtering the training data.')
    parser.add_argument('--val_filing_start_date', type=str, default='', help='Start date for filtering the training data.')
    parser.add_argument('--val_filing_end_date', type=str, default='', help='End date for filtering the validation data.')
    parser.add_argument('--use_wsampler', action='store_true', help='Use a weighted sampler (for the training set).')
    parser.add_argument('--val_set_balancer', action='store_true', help='Use a balanced set for validation? That is, do you want the same number of classes of examples in the validation set.')
    parser.add_argument('--uniform_split', default=True, help='Uniformly split the data into training and validation sets.')
    parser.add_argument('--num_proc', default=8, help='Number of processors to use for preprocessing data')
    parser.add_argument('--validation', default=False, help='Perform only validation/inference. (No performance evaluation on the training data necessary).')
    parser.add_argument('--max_length', type=int, default=4096, help='The maximum total input sequence length after tokenization. Sequences longer than this number will be trunacated.')

    # Training
    parser.add_argument('--accumulation_steps', default=0, help='Num steps to accum gradient')
    parser.add_argument('--train_from_scratch', action='store_true', help='Train the model from the scratch.')
    parser.add_argument('--validation', default=True, help='Perform only validation/inference. (No performance evaluation on the training data necessary).')
    parser.add_argument('--batch_size', type=dict, default={'train':8, 'validation':48}, help='Batch size.')
    parser.add_argument('--epoch_n', type=int, default=2, help='Number of epochs (for training).')
    parser.add_argument('--val_every', type=int, default=2000, help='Number of iterations we should take to perform validation.')
    parser.add_argument('--validate_training_every', type=int, default=8500, help='Number of iterations we should take to perform training validation.')
    parser.add_argument('--lr', type=float, default=2e-5, help='Model learning rate.')
    parser.add_argument('--wandb', action='store_true', help='Use wandb.')
    parser.add_argument('--wandb_name', type=str, default=None, help='wandb project name.')
    parser.add_argument('--use_scheduler', action='store_true', help='Use a scheduler.')
    parser.add_argument('--tensorboard', default=True, help='Use tensorboard.')
    parser.add_argument('--handle_skew_data', type=bool, default=True, help='Add class weights based on their fraction of the total data')
    parser.add_argument('--continue_training', type=bool, default=False, help='Load weights and continue training')
    parser.add_argument('--linear_probe', type=bool, default=False, help='Load weights and continue training')
    parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon value for the learning rate.')

    # Saving purposes
    parser.add_argument('--filename', type=str, default=None, help='Name of the results file to be saved.')
    

    mistral_model_name = "distilbert-base-uncased"
    # Model related params
    model_path = "CS224N_models/distilbert-base-uncased/claims_distilbert-base-uncased_2_8_2e-05_512_False_all_False_date_3_2_hr_22/epoch_"
    parser.add_argument('--model_name', type=str, default=mistral_model_name, help='Name of the model.')
    parser.add_argument('--model_path', type=str, default=model_path + "model", help='(Pre-trained) model path.')
    parser.add_argument('--tokenizer_path', type=str, default=model_path + "tokenizer", help='(Pre-trained) tokenizer path.')
    parser.add_argument('--save_path', type=str, default="CS224N_models", help='The path where the model is going to be saved.')
    # parser.add_argument('--save_path', type=str, default=None, help='The path where the model is going to be saved.')

    parser.add_argument('--tokenizer_save_path', type=str, default=None, help='The path where the tokenizer is going to be saved.')
    parser.add_argument('--max_length', type=int, default=512, help='The maximum total input sequence length after tokenization. Sequences longer than this number will be trunacated.')



    # Parse args
    args = parser.parse_args()

    device = "cuda"
    model_id = "mistralai/Mistral-7B-v0.1"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )

    peft_config = LoraConfig(
        r=script_args.lora_r,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="SEQ_CLS",
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        inference_mode=False
    )
    model = get_peft_model(model, peft_config)


        # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right")
    if getattr(tokenizer, "pad_token_id") is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id


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
        dataset_dict[name] = dataset_dict[name].filter(lambda e: e['labels'] <= 1)
        dataset_dict[name] = dataset_dict[name].remove_columns(set(dataset_dict[name].column_names) - set(["labels", args.section]))


            # Load the dataset
    data_loaders = create_dataset(
        args = args, 
        dataset_dict = dataset_dict, 
        tokenizer = tokenizer, 
        section = args.section,
        )
    del dataset_dict

    # Instantiate dataloaders.
    train_dataloader = data_loaders[0]
    eval_dataloader = data_loaders[1]

    # %%



    # %%
    optimizer = AdamW(params=model.parameters(), lr=script_args.lr, weight_decay=script_args.weight_decay)

    # Instantiate scheduler
    lr_scheduler = get_constant_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=script_args.warmup_ratio * (len(train_dataloader) * args.epoch_n),
        num_training_steps=(len(train_dataloader) * args.epoch_n),
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
    for epoch in range(args.epoch_n):
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader)):
            batch.to(device, non_blocking=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), script_args.max_grad_norm)
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

