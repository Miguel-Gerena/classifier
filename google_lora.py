import os
if os.getlogin() == "darke":
    PATH =  "D:/classes/cache/huggingface/hub"
    os.environ['HF_HOME'] = PATH
    os.environ['HF_DATASETS_CACHE'] = PATH

from dataclasses import dataclass, field
from typing import Optional

import json
import torch
import argparse

from transformers import AutoTokenizer, HfArgumentParser, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, is_bitsandbytes_available
from transformers import AutoModelForSequenceClassification, AutoConfig
from transformers import MistralForSequenceClassification, MistralConfig
from data_handling import map_decision_to_string, create_model_and_tokenizer, dataset_statistics, measure_accuracy, create_dataset


from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer

@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=4)
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=0.3)
    weight_decay: Optional[int] = field(default=0.001)
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=8)
    max_seq_length: Optional[int] = field(default=4096)
    model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        }
    )

    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=True,
        metadata={"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    use_flash_attention_2: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash Attention 2."},
    )
    optim: Optional[str] = field(
        default="paged_adamw_32bit",
        metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
    warmup_ratio: float = field(default=0.03, metadata={"help": "Fraction of steps to do a warmup for"})
    save_steps: int = field(default=1, metadata={"help": "Save checkpoint every X updates steps."})
    logging_steps: int = field(default=1, metadata={"help": "Log every X updates steps."})
    output_dir: str = field(
        default="./results",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
 

def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    script_args.max_steps=315984//(script_args.gradient_accumulation_steps * script_args.per_device_train_batch_size)  # how many batches with a batch size of 4


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
    parser.add_argument('--handle_skew_data', type=bool, default=True, help='Add class weights based on their fraction of the total data')
    parser.add_argument('--continue_training', type=bool, default=True, help='Load weights and continue training')


    # Parse args
    args = parser.parse_args()

    # Load the GG model - this is the local one, update it to the one on the Hub
    model_id = "mistralai/Mistral-7B-v0.1"

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    # Load model
    CLASSES = 2
    config = MistralConfig.from_pretrained(
        model_id,
        num_labels=CLASSES, 
        output_hidden_states=False, 
        quantization_config=quantization_config, 
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa" if not script_args.use_flash_attention_2 else "flash_attention_2")

    model = MistralForSequenceClassification._from_config(config)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.max_length = script_args.max_seq_length
    tokenizer.model_max_length = script_args.max_seq_length
    tokenizer.padding_side = 'right'
    model.config.pad_token_id = model.config.eos_token_id
    args.batch_size={'train':script_args.per_device_train_batch_size, 'validation':script_args.per_device_eval_batch_size}

    lora_config = LoraConfig(
        r=script_args.lora_r,
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="SEQ_CLS",
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout
    )
    output_dir = f"CS224N_models/{model_id.split("/")[-1]}/{args.section}__{script_args.per_device_train_batch_size * script_args.gradient_accumulation_steps}_{script_args.learning_rate}_{args.continue_training}_{args.dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    write_file = open(f"{output_dir}/settings.txt", "w")

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
        dataset_dict[name] = dataset_dict[name].rename_column("output", "labels")
        dataset_dict[name] = dataset_dict[name].remove_columns(set(dataset_dict[name].column_names) - set(["labels", args.section]))

    # def collate_fn(examples):
    #     return tokenizer.pad(examples, padding="longest", return_tensors="pt")
    #     # Load the dataset
    # data_loaders = create_dataset(
    #     args = args, 
    #     dataset_dict = dataset_dict, 
    #     tokenizer = tokenizer, 
    #     section = args.section,
    #     collate_fn=collate_fn
    #     )
    # del dataset_dict

    with open(f"{output_dir}/arguments.json", "w") as file:
        json.dump(script_args.__dict__, file)
        json.dump(args.__dict__, file)

    # train_label_stats = dataset_statistics( data_loaders[0])
    # val_label_stats = dataset_statistics( data_loaders[1])
    # print(f'*** Training set label statistics: {train_label_stats}')
    # print(f'*** Validation set label statistics: {val_label_stats}')
    # write_file.write(f'*** Training set label statistics: {train_label_stats}\n')
    # write_file.write(f'*** Validation set label statistics: {val_label_stats}\n\n')
    write_file.close()

    training_arguments = TrainingArguments(
        output_dir="./mistral",
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        optim=script_args.optim,
        save_steps=script_args.save_steps,
        logging_steps=script_args.logging_steps,
        learning_rate=script_args.learning_rate,
        max_grad_norm=script_args.max_grad_norm,
        max_steps=script_args.max_steps,
        warmup_ratio=script_args.warmup_ratio,
        lr_scheduler_type=script_args.lr_scheduler_type,
        gradient_checkpointing=script_args.gradient_checkpointing,
        fp16=script_args.fp16,
        bf16=script_args.bf16,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset_dict["train"],
        peft_config=lora_config,
        dataset_text_field="claims",
        tokenizer=tokenizer,
        max_seq_length=script_args.max_seq_length,
        dataset_num_proc=8
    )

    trainer.train()

if __name__ == "__main__":
    main()