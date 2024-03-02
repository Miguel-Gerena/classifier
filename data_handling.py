
import torch
import random
import numpy as np
import collections
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm


# Good old Transformer models
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer, GPT2Config
from transformers import LongformerForSequenceClassification, LongformerTokenizer, LongformerConfig
from transformers import PreTrainedTokenizerFast
from transformers import BitsAndBytesConfig


# Import the sklearn Multinomial Naive Bayes
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

# Simple LSTM, CNN, and Logistic regression models
from models import BasicCNNModel, BigCNNModel, LogisticRegression

# Tokenizer-releated dependencies
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

# Confusion matrix
from sklearn.metrics import confusion_matrix, f1_score

import lora, mistal7b

# wandb
try:
    import wandb
except ImportError:
    wandb = None

# PyTorch


# Fixing the random seeds
RANDOM_SEED = 1729
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# Number of classes (ACCEPTED and REJECTED)
CLASSES = 2
CLASS_NAMES = [i for i in range(CLASSES-1, -1, -1)]

# Create model and tokenizer
def create_model_and_tokenizer(args, train_from_scratch=False, model_name='bert-base-uncased',
                             dataset=None,  vocab_size=10000, max_length=512):

    if args.validation or args.continue_training:
        if model_name == 'distilbert-base-uncased':
            if args.model_path:
                print("**" * 4, "Loading Pre-trained weights", "**" * 4)
                tokenizer = DistilBertTokenizer.from_pretrained(args.tokenizer_path) 
                model = DistilBertForSequenceClassification.from_pretrained(args.model_path)
            else:
                config = DistilBertConfig(num_labels=CLASSES, output_hidden_states=False) 
                tokenizer = DistilBertTokenizer.from_pretrained(model_name, do_lower_case=True)
                model = DistilBertForSequenceClassification(config=config)
  
        elif model_name == 'mistralai/Mistral-7B-v0.1':
            config = AutoConfig.from_pretrained(model_name, num_labels=CLASSES, output_hidden_states=False)
            model = AutoModelForSequenceClassification.from_config(config)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif model_name == "google/gemma-2b":
            with open("./.env") as file:
                for line in file:
                    token = line
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            config = AutoConfig.from_pretrained(model_name, num_labels=CLASSES, output_hidden_states=False, token=token, quantization_config=quantization_config, device_map="auto")
            model = AutoModelForSequenceClassification.from_config(config)
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
            tokenizer.pad_token_id = tokenizer.eos_token_id

        else:
            raise NotImplementedError
        # This step is actually important.
        tokenizer.max_length = max_length
        tokenizer.model_max_length = max_length
    else:
        # Train from scratch
        if train_from_scratch:
            if model_name == 'bert-base-uncased':
                config = BertConfig(num_labels=CLASSES, output_hidden_states=False) 
                tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
                model = BertForSequenceClassification(config=config)
            elif model_name == 'distilbert-base-uncased':
                config = DistilBertConfig(num_labels=CLASSES, output_hidden_states=False) 
                tokenizer = DistilBertTokenizer.from_pretrained(model_name, do_lower_case=True)
                model = DistilBertForSequenceClassification(config=config)
            elif model_name == 'roberta-base':
                config = RobertaConfig(num_labels=CLASSES, output_hidden_states=False) 
                tokenizer = RobertaTokenizer.from_pretrained(model_name, do_lower_case=True)
                model = RobertaForSequenceClassification(config=config)
            elif model_name == 'mistralai/Mistral-7B-v0.1':
                config = AutoConfig.from_pretrained(model_name, num_labels=CLASSES, output_hidden_states=False)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
            else:
                raise NotImplementedError()

        # Finetune
        else:
            if model_name in ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base', 'gpt2', 'allenai/longformer-base-4096', 'mistralai/Mistral-7B-v0.1', "google/gemma-2b", "google/gemma-7b"]:
                config = AutoConfig.from_pretrained(model_name, num_labels=CLASSES, output_hidden_states=False)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if model_name in ['gpt2', 'mistralai/Mistral-7B-v0.1', "google/gemma-7b", "google/gemma-2b"]:
                    tokenizer.pad_token = tokenizer.eos_token
                tokenizer.max_length = max_length
                tokenizer.model_max_length = max_length
                if model_name == 'mistralai/Mistral-7B-v0.1':
                    rank = 64
                    model = mistal7b.CustomizedMistralModel(model_name=model_name, rank=rank, num_labels=CLASSES)
                else:
                    model = AutoModelForSequenceClassification.from_config(config=config)
            
    if model in ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base', 'gpt2', 'allenai/longformer-base-4096', 'mistralai/Mistral-7B-v0.1', "google/gemma-2b", "google/gemma-7b"]:
        print(f'Model name: {model_name} \nModel params: {model.num_parameters()}')
    else:
        print(model)
    return tokenizer, dataset, model, vocab_size


# Map decision2string
def map_decision_to_string(example):
    decision_to_str = {
    'REJECTED': 0, 
    'ACCEPTED': 1, 
    'PENDING': 2, 
    'CONT-REJECTED': 3, 
    'CONT-ACCEPTED': 4, 
    'CONT-PENDING': 5
}
    return {'output': decision_to_str[example['decision']]}

# Create dataset
def create_dataset(args, dataset_dict, tokenizer, section='abstract', use_wsampler=True, write_file=None):
    data_loaders = []
    for name in ['train', 'validation']:
        # Skip the training set if we are doing only inference
        if args.validation and name=='train':
            data_loaders.append(None)
        else:
            dataset = dataset_dict[name]
            
            print('*** Tokenizing...')

            def combine(data):
                return {"examiner": f"Examiner id: {data['examiner_id'][:-2]} " + data['claims']}
            
            if section == "examiner":
                dataset = dataset.map(combine, num_proc=args.num_proc)

            cols_keep = [section, "output", "decision"]

            for col in dataset.column_names:
                if col not in cols_keep:
                    dataset = dataset.remove_columns(col)
            

            dataset = dataset.map(
                lambda e: tokenizer(e[section], truncation=True, padding='max_length'),
                batched=True, num_proc=args.num_proc)
                

            # Set the dataset format
            dataset.set_format(type='torch', 
                columns=['input_ids', 'attention_mask', 'output'])

            # Check if we are using a weighted sampler for the training set
            if use_wsampler and name == 'train':
                # https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/10
                target = dataset['output']
                class_sample_count = torch.tensor([(target == t).sum() for t in torch.unique(target, sorted=True)])
                weight = 1. / class_sample_count.float()
                samples_weight = torch.tensor([weight[t] for t in target])
                sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)
                data_loaders.append(DataLoader(dataset, batch_size=args.batch_size, sampler=sampler))
                print(f'*** Set: {name} (using a weighted sampler).')
                print(f'*** Weights: {weight}')
                if write_file:
                    write_file.write(f'*** Set: {name} (using a weighted sampler).\n')
                    write_file.write(f'*** Weights: {weight}\n')
            else:
                data_loaders.append(DataLoader(dataset, batch_size=args.batch_size[name], shuffle=(name=='train')))
    return data_loaders


# Return label statistics of the dataset loader
def dataset_statistics(dataset_loader):
    label_stats = collections.Counter()
    for i, batch in enumerate(tqdm(dataset_loader)):
        _, decisions = batch['input_ids'], batch['output']
        labels = decisions.cpu().numpy().flatten()
        label_stats += collections.Counter(labels)
    return label_stats

# Calculate TOP1 accuracy
def measure_accuracy(preds, labels):
    correct = np.sum(preds == labels)
    c_matrix = confusion_matrix(labels, preds, labels=CLASS_NAMES)
    f1 = f1_score(labels, preds, labels=CLASS_NAMES)
    return correct, len(labels), c_matrix, f1

# Convert ids2string
def convert_ids_to_string(tokenizer, input):
    return ' '.join(tokenizer.convert_ids_to_tokens(input)) # tokenizer.decode(input)
