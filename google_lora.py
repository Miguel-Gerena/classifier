from dataclasses import dataclass, field
from typing import Optional

import torch

from transformers import AutoTokenizer, HfArgumentParser, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoConfig

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
    max_seq_length: Optional[int] = field(default=2048)
    model_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        }
    )
    dataset_name: Optional[str] = field(
        default="stingning/ultrachat",
        metadata={"help": "The preference dataset to use."},
    )
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=False,
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
    max_steps: int = field(default=1000, metadata={"help": "How many optimizer update steps to take"})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Fraction of steps to do a warmup for"})
    save_steps: int = field(default=10, metadata={"help": "Save checkpoint every X updates steps."})
    logging_steps: int = field(default=10, metadata={"help": "Log every X updates steps."})
    output_dir: str = field(
        default="./results",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]



# Load the GG model - this is the local one, update it to the one on the Hub
model_id = "google/gemma-7b"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)
CLASSES = 2
# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, 
    num_labels=CLASSES, 
    output_hidden_states=False,
    quantization_config=quantization_config, 
    torch_dtype=torch.float32,
    attn_implementation="sdpa" if not script_args.use_flash_attention_2 else "flash_attention_2"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id

lora_config = LoraConfig(
    r=script_args.lora_r,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="SEQ_CLS",
    lora_alpha=script_args.lora_alpha,
    lora_dropout=script_args.lora_dropout
)

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

print(f'*** CPC Label: {cat_label}') 
print(f'*** Section: {args.section}')
print(f'*** Vocabulary: {args.vocab_size}')

if write_file:
    write_file.write(f'*** date time: {now.month}_{now.day}_hr_{now.hour}\n')
    write_file.write(f'*** CPC Label: {cat_label}\n')
    write_file.write(f'*** Section: {args.section}\n')
    write_file.write(f'*** Vocabulary: {args.vocab_size}\n')
    write_file.write(f'*** args: {args}\n\n')


    # Load the dataset
data_loaders = create_dataset(
    args = args, 
    dataset_dict = dataset_dict, 
    tokenizer = tokenizer, 
    section = args.section,
    use_wsampler=args.use_wsampler,
    write_file=write_file
    )
del dataset_dict

train_dataset = load_dataset(script_args.dataset_name, split="train[:5%]")

# TODO: make that configurable
YOUR_HF_USERNAME = xxx
output_dir = f"{YOUR_HF_USERNAME}/gemma-qlora-ultrachat"

training_arguments = TrainingArguments(
    output_dir=output_dir,
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
    train_dataset=data_loaders[0],
    eval_dataset=data_loaders[1],
    peft_config=lora_config,
    packing=script_args.packing,
    dataset_text_field=args.section,
    tokenizer=tokenizer,
    max_seq_length=script_args.max_seq_length,
)

trainer.train()