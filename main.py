# Standard libraries and dependencies
import os
from unittest.mock import DEFAULT
if os.getlogin() == "darke":
    PATH =  "D:/classes/cache/huggingface/hub"
    os.environ['TRANSFORMERS_CACHE'] = PATH
    os.environ['HF_HOME'] = PATH
    os.environ['HF_DATASETS_CACHE'] = PATH
    
import argparse
import random
import numpy as np
import collections
from tqdm import tqdm
import datetime
import json
import pandas as pd


# wandb
try:
    import wandb
except ImportError:
    wandb = None

# PyTorch
import torch
from torch.utils import tensorboard
from torch.utils.data import DataLoader, WeightedRandomSampler

# Hugging Face datasets
from datasets import load_dataset

# Good old Transformer models
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer, GPT2Config
from transformers import LongformerForSequenceClassification, LongformerTokenizer, LongformerConfig
from transformers import PreTrainedTokenizerFast

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

# For scheduling 
from transformers import get_linear_schedule_with_warmup

# Confusion matrix
from sklearn.metrics import confusion_matrix, f1_score

#additional metrics
from torcheval.metrics import BinaryAccuracy, BinaryF1Score, BinaryAUPRC

# Fixing the random seeds
RANDOM_SEED = 1729
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Number of classes (ACCEPTED and REJECTED)
CLASSES = 2
CLASS_NAMES = [i for i in range(CLASSES-1, -1, -1)]

# Create a BoW (Bag-of-Words) representation
def text2bow(input, vocab_size):
    arr = []
    for i in range(input.shape[0]):
        query = input[i]
        features = [0] * vocab_size
        for j in range(query.shape[0]):
            features[query[j]] += 1 # todo: for Multinomial (initially +1)
        arr.append(features)
    return np.array(arr)

# Create model and tokenizer
def create_model_and_tokenizer(args, train_from_scratch=False, model_name='bert-base-uncased',
                             dataset=None, section='abstract', vocab_size=10000, embed_dim=200, n_classes=CLASSES, max_length=512):
    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

    if args.validation:
        if model_name == 'distilbert-base-uncased':
            if args.model_path:
                tokenizer = DistilBertTokenizer.from_pretrained(args.tokenizer_path) 
                model = DistilBertForSequenceClassification.from_pretrained(args.model_path)
            else:
                config = DistilBertConfig(num_labels=CLASSES, output_hidden_states=False) 
                tokenizer = DistilBertTokenizer.from_pretrained(model_name, do_lower_case=True)
                model = DistilBertForSequenceClassification(config=config)
            # This step is actually important.
            tokenizer.max_length = max_length
            tokenizer.model_max_length = max_length
        else:
            raise NotImplementedError
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
            elif model_name == 'gpt2':
                config = GPT2Config(num_labels=CLASSES, output_hidden_states=False) 
                tokenizer = GPT2Tokenizer.from_pretrained(model_name, do_lower_case=True)
                model = GPT2ForSequenceClassification(config=config)
            elif model_name == 'allenai/longformer-base-4096':
                config = LongformerConfig(num_labels=CLASSES, output_hidden_states=False) 
                tokenizer = LongformerTokenizer.from_pretrained(model_name, do_lower_case=True)
                model = LongformerForSequenceClassification(config=config)
            else:
                raise NotImplementedError()

        # Finetune
        else:
            if model_name in ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base', 'gpt2', 'allenai/longformer-base-4096']:
                config = AutoConfig.from_pretrained(model_name, num_labels=CLASSES, output_hidden_states=False)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if model_name == 'gpt2':
                    tokenizer.pad_token = tokenizer.eos_token
                tokenizer.max_length = max_length
                tokenizer.model_max_length = max_length
                model = AutoModelForSequenceClassification.from_config(config=config)
            elif model_name in ['lstm', 'cnn', 'big_cnn', 'naive_bayes', 'logistic_regression']:
                # Word-level tokenizer
                tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
                # Normalizers
                tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
                # World-level trainer
                trainer = WordLevelTrainer(vocab_size=vocab_size, min_frequency=3, show_progress=True, 
                    special_tokens=special_tokens)
                # Whitespace (pre-tokenizer)
                tokenizer.pre_tokenizer = Whitespace()
                # Train from iterator
                tokenizer.train_from_iterator(dataset['train'][section], trainer=trainer)                
                # Update the vocab size
                vocab_size = tokenizer.get_vocab_size()
                # [PAD] idx
                pad_idx = tokenizer.encode('[PAD]').ids[0]

                # Currently the call method for WordLevelTokenizer is not working.
                # Using this temporary method until the tokenizers library is updated.
                # Not a fan of this method, but this is the best we have right now (sad face).
                # Based on https://github.com/huggingface/transformers/issues/7234#issuecomment-720092292
                tokenizer.enable_padding(pad_type_id=pad_idx)
                tokenizer.pad_token = '[PAD]'
                args.vocab_size = vocab_size

                if args.model_name != 'naive_bayes': # CHANGE 'naive_bayes' (shannon)
                    tokenizer.model_max_length = max_length
                    tokenizer.max_length = max_length
                tokenizer.save("temp_tokenizer.json") 
                if args.tokenizer_save_path:
                    print('*** Saving the tokenizer...')
                    tokenizer.save(f"{args.tokenizer_save_path}")
                tokenizer = PreTrainedTokenizerFast(tokenizer_file="temp_tokenizer.json")

                if args.model_name != 'naive_bayes': # CHANGE 'naive_bayes'
                    tokenizer.model_max_length = max_length
                    tokenizer.max_length = max_length
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    tokenizer.pad_token = '[PAD]'
                    tokenizer.add_special_tokens({'sep_token': '[SEP]'})
                    tokenizer.sep_token = '[SEP]'

                model = None
                if model_name == 'logistic_regression':
                    model = LogisticRegression(vocab_size=vocab_size, embed_dim=embed_dim, n_classes=n_classes, pad_idx=pad_idx)
                elif model_name == 'cnn':
                    model = BasicCNNModel(vocab_size=vocab_size, embed_dim=embed_dim, pad_idx=pad_idx, n_classes=n_classes, n_filters=args.n_filters, filter_sizes=args.filter_sizes[0], dropout=args.dropout)
                elif model_name == 'big_cnn':
                    model = BigCNNModel(vocab_size=vocab_size, embed_dim=embed_dim, pad_idx=pad_idx, n_classes=n_classes, n_filters=args.n_filters, filter_sizes=args.filter_sizes, dropout=args.dropout)
            else:
                raise NotImplementedError()
            
    if model in ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base', 'gpt2', 'allenai/longformer-base-4096']:
        print(f'Model name: {model_name} \nModel params: {model.num_parameters()}')
    else:
        print(model)
    return tokenizer, dataset, model, vocab_size

# For filtering out CONT-apps and pending apps
decision_to_str = {
    'REJECTED': 0, 
    'ACCEPTED': 1, 
    'PENDING': 2, 
    'CONT-REJECTED': 3, 
    'CONT-ACCEPTED': 4, 
    'CONT-PENDING': 5
}

# Map decision2string
def map_decision_to_string(example):
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

            # Tokenize the input
            cols = pd.DataFrame(dataset)
            if section == "combined":
                cols["combined"] = cols["abstract"] + cols["claims"]
                dataset = dataset.add_column(name="combined", column=cols["combined"])
            print(f'*** Longest input in {name} dataset abstract: {cols["abstract"].str.len().max()}')
            write_file.write(f'*** Longest input in {name} dataset abstract: {cols["abstract"].str.len().max()}\n')
            print(f'*** Longest input in {name} dataset claims: {cols["claims"].str.len().max()}')
            write_file.write(f'*** Longest input in {name} dataset claims: {cols["claims"].str.len().max()}\n')
            print(f'*** Longest input in used {name} dataset {args.section}: {cols[f"{section}"].str.len().max()}')
            write_file.write(f'*** Longest input in used {name} dataset {args.section}: {cols[f"{section}"].str.len().max()}')
            del cols

            dataset = dataset.map(
                lambda e: tokenizer(e[section], truncation=True, padding='max_length'),
                batched=True)
                

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
                data_loaders.append(DataLoader(dataset, batch_size=args.batch_size, shuffle=(name=='train')))
    return data_loaders


# Return label statistics of the dataset loader
def dataset_statistics(args, dataset_loader, tokenizer):
    label_stats = collections.Counter()
    for i, batch in enumerate(tqdm(dataset_loader)):
        inputs, decisions = batch['input_ids'], batch['output']
        labels = decisions.cpu().numpy().flatten()
        label_stats += collections.Counter(labels)
    return label_stats

# Calculate TOP1 accuracy
def measure_accuracy(preds, labels, torch_metrics = None):
    correct = np.sum(preds == labels)
    c_matrix = confusion_matrix(labels, preds, labels=CLASS_NAMES)
    f1 = f1_score(labels, preds, labels=CLASS_NAMES)
    return correct, len(labels), c_matrix, f1

# Convert ids2string
def convert_ids_to_string(tokenizer, input):
    return ' '.join(tokenizer.convert_ids_to_tokens(input)) # tokenizer.decode(input)

# Evaluation procedure (for the neural models)
def validation(args, val_loader, model, criterion, device, name='validation', write_file=None, tensorboard_writer=None, step = 0, use_torch_metrics=True):
    model.eval()
    total_loss = 0.
    total_correct = 0
    total_sample = 0
    total_confusion = np.zeros((CLASSES, CLASSES))

    if use_torch_metrics:
        torch_metrics = {}
        torch_metrics["acc"] = BinaryAccuracy()
        torch_metrics["f1"] = BinaryF1Score()
        torch_metrics["auc"] = BinaryAUPRC()

        # Loop over the examples in the evaluation set
    for i, batch in enumerate(tqdm(val_loader)):
        inputs, decisions = batch['input_ids'], batch['output']
        inputs = inputs.to(device)
        decisions = decisions.to(device)
        with torch.no_grad():
            if args.model_name in ['lstm', 'cnn', 'big_cnn', 'naive_bayes', 'logistic_regression']:
                outputs = model(input_ids=inputs)
            else:
                outputs = model(input_ids=inputs, labels=decisions).logits
        loss = criterion(outputs, decisions) 
        logits = outputs 
        total_loss += loss.cpu().item()

        preds = torch.argmax(logits, axis=1).flatten()
        labels = decisions.flatten()

        correct_n, sample_n, c_matrix, f1 = measure_accuracy(preds.cpu().numpy(), labels.cpu().numpy())
        total_confusion += c_matrix
        total_correct += correct_n
        total_sample += sample_n

        torch_metrics["acc"] = torch_metrics["acc"].update(preds, labels)
        torch_metrics["f1"] = torch_metrics["acc"].update(preds, labels)
        torch_metrics["auc"] = torch_metrics["acc"].update(preds, labels)

    mean_loss = total_loss/total_sample
    acc = total_correct/total_sample
    total_f1 = torch_metrics["f1"].compute()

    # Print the performance of the model on the validation set 
    print(f'*** Accuracy on the {name} set: {acc}')
    print(f'*** Confusion matrix:\n{total_confusion}')

    if args.tensorboard:
        tensorboard_writer.add_scalar(f'val/{name}_mean_loss', mean_loss, step)
        tensorboard_writer.add_scalar(f'val/{name}_acc', acc, step)   

        tensorboard_writer.add_scalar(f'val/{name}_torch_acc', torch_metrics["acc"].compute(), step)   
        tensorboard_writer.add_scalar(f'val/{name}_torch_f1', total_f1, step)   
        tensorboard_writer.add_scalar(f'val/{name}_torch_auc', torch_metrics["auc"].compute(), step)   


    if write_file:
        write_file.write(f'*** Accuracy on the {name} set: {total_correct/total_sample}\n')
        write_file.write(f'*** Confusion matrix:\n{total_confusion}\n')
        write_file.flush()

    return mean_loss, float(acc) * 100., total_f1


# Training procedure (for the neural models)
def train(args, data_loaders, epoch_n, model, optim, scheduler, criterion, device, write_file=None, tensorboard_writer=None):
    print('\n>>>Training starts...')
    if write_file:
        write_file.write(f'\n>>>Training starts...\n')
    # Training mode is on
    model.train()
    # Best validation set accuracy so far.
    best_val_acc = 0
    best_f1 = 0
    validation(args,data_loaders[0], model, criterion, device, name='train', tensorboard_writer=tensorboard_writer, step=0)
    validation(args,data_loaders[1], model, criterion, device, name='validation', tensorboard_writer=tensorboard_writer, step=0)
    for epoch in range(epoch_n):
        total_train_loss = 0.
        # Loop over the examples in the training set.
        for i, batch in enumerate(tqdm(data_loaders[0])):
            inputs, decisions = batch['input_ids'], batch['output']
            inputs = inputs.to(device, non_blocking=True)
            decisions = decisions.to(device, non_blocking=True)
            
            # Forward pass
            if args.model_name in ['lstm', 'cnn', 'big_cnn', 'logistic_regression']:
                outputs = model(input_ids=inputs)
            else:
                outputs = model(input_ids=inputs, labels=decisions).logits
            loss = criterion(outputs, decisions) #outputs.logits
            total_train_loss += loss.cpu().item()

            # Backward pass
            loss.backward()
            optim.step()
            if scheduler:
                scheduler.step()
            optim.zero_grad()
            
            # wandb (optional)
            if args.wandb:
                wandb.log({'Training Loss': loss})
            
            if args.tensorboard:
                tensorboard_writer.add_scalar('train/loss', loss.item(), epoch * len(data_loaders[0]) + i)
                tensorboard_writer.add_scalar('train/mean_loss', total_train_loss / (i if i > 0 else 1), epoch * len(data_loaders[0]) + i)


            # Print the loss every val_every step
            if (epoch * len(data_loaders[0]) + i) % args.val_every == 0 and i !=0:
                print(f'*** Loss: {loss}')
                print(f'*** Input: {convert_ids_to_string(tokenizer, inputs[0])}')
                if write_file:
                    write_file.write(f'\nEpoch: {epoch}, Step: {i}\n')
                # Get the performance of the model on the validation set
                mean_loss, val_acc, f1_acc = validation(args, data_loaders[1], model, criterion, device, write_file=write_file, tensorboard_writer=tensorboard_writer, step=epoch * len(data_loaders[0]) + i)
                model.train()

                if args.wandb:
                    wandb.log({'Validation Accuracy': val_acc})

                if best_val_acc < val_acc:
                    best_val_acc = val_acc
                    # Save the model if a save directory is specified
                    if args.save_path:
                        # If the model is a Transformer architecture, make sure to save the tokenizer as well
                        if args.model_name in ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base', 'gpt2', 'allenai/longformer-base-4096']:
                            model.save_pretrained(args.save_path + 'model')
                            tokenizer.save_pretrained(args.save_path + 'tokenizer')
                        else:
                            torch.save(model.state_dict(), args.save_path)


        if (epoch * len(data_loaders[0]) + i) % args.validate_training_every_epoch == 0 and i !=0:
            validation(args, data_loaders[0], model, criterion, device, name='train', tensorboard_writer=tensorboard_writer, step=epoch * len(data_loaders[0]) + i)

    # Training is complete!
    print(f'\n ~ The End ~')
    if write_file:
        write_file.write('\n ~ The End ~\n')
    
    # Final evaluation on the validation set
    _, val_acc, f1_acc = validation(args, data_loaders[1], model, criterion, device, name='validation', write_file=write_file, tensorboard_writer=tensorboard_writer)
    if best_val_acc < val_acc:
        best_val_acc = val_acc
        
        # Save the best model so fare
        if args.save_path:
            if args.model_name in ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base', 'gpt2', 'allenai/longformer-base-4096']:
                model.save_pretrained(args.save_path + 'model')
            else:
                torch.save(model.state_dict(), args.save_path)

    if best_f1 < f1_acc:
        best_f1 = f1_acc
        # Save the model if a save directory is specified
        if args.save_path:
            # If the model is a Transformer architecture, make sure to save the tokenizer as well
            if args.model_name in ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base', 'gpt2', 'allenai/longformer-base-4096']:
                model.save_pretrained(args.save_path + "f1_model")
                tokenizer.save_pretrained(args.save_path + 'f1_tokenizer')
            else:
                torch.save(model.state_dict(), args.save_path)
    
    
    # Additionally, print the performance of the model on the training set if we were not doing only inference
    if not args.validation:
        validation(args,data_loaders[0], model, criterion, device, name='train', tensorboard_writer=tensorboard_writer, step=0)
    
    # Print the highest accuracy score obtained by the model on the validation set
    print(f'*** Highest accuracy on the validation set: {best_val_acc}.')
    print(f'*** Highest f1 accuract on the validation set: {best_f1}.')

    if write_file:
        write_file.write(f'\n*** Highest accuracy on the validation set: {best_val_acc}.')
        write_file.write(f'\n*** Highest f1 accuracy on the validation set: {best_f1}.')



 

# Evaluation procedure (for the Naive Bayes models)
def validation_naive_bayes(data_loader, model, vocab_size, name='validation', write_file=None, pad_id=-1):
    total_loss = 0.
    total_correct = 0
    total_sample = 0
    total_confusion = np.zeros((CLASSES, CLASSES))
    
    # Loop over all the examples in the evaluation set
    for i, batch in enumerate(tqdm(data_loader)):
        input, label = batch['input_ids'], batch['output']
        input = text2bow(input, vocab_size)
        input[:, pad_id] = 0
        logit = model.predict_log_proba(input)
        label = np.array(label.flatten()) 
        correct_n, sample_n, c_matrix = measure_accuracy(logit, label)
        total_confusion += c_matrix
        total_correct += correct_n
        total_sample += sample_n
    print(f'*** Accuracy on the {name} set: {total_correct/total_sample}')
    print(f'*** Confusion matrix:\n{total_confusion}')
    if write_file:
        write_file.write(f'*** Accuracy on the {name} set: {total_correct/total_sample}\n')
        write_file.write(f'*** Confusion matrix:\n{total_confusion}\n')
    return total_loss, float(total_correct/total_sample) * 100.


# Training procedure (for the Naive Bayes models)
def train_naive_bayes(data_loaders, tokenizer, vocab_size, version='Bernoulli', alpha=1.0, write_file=None, np_filename=None):
    pad_id = tokenizer.encode('[PAD]') # NEW
    print(f'Training a {version} Naive Bayes classifier (with alpha = {alpha})...')
    write_file.write(f'Training a {version} Naive Bayes classifier (with alpha = {alpha})...\n')

#     # Bernoulli or Multinomial?
#     if version == 'Bernoulli':
#         model = BernoulliNB(alpha=alpha) 
#     elif version == 'Multinomial':
#         model = MultinomialNB(alpha=alpha) 
    
#     # Loop over all the examples in the training set
#     for i, batch in enumerate(tqdm(data_loaders[0])):
#         input, decision = batch['input_ids'], batch['output']
#         input = text2bow(input, vocab_size) # change text2bow(input[0], vocab_size)
#         input[:, pad_id] = 0 # get rid of the paddings
#         label = np.array(decision.flatten())
#         # Using "partial fit", instead of "fit", to avoid any potential memory problems
#         # model.partial_fit(np.array([input]), np.array([label]), classes=CLASS_NAMES)
#         model.partial_fit(input, label, classes=CLASS_NAMES)
    
#     print('\n*** Accuracy on the training set ***')
#     validation_naive_bayes(data_loaders[0], model, vocab_size, 'training', write_file, pad_id)
#     print('\n*** Accuracy on the validation set ***')
#     validation_naive_bayes(data_loaders[1], model, vocab_size, 'validation', write_file, pad_id)
    
#     # Save the log probabilities if np_filename is specified
#     if np_filename:
#         np.save(f'{np_filename}.npy', np.array(model.feature_log_prob_))

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    parser = argparse.ArgumentParser()
    
    # Dataset
    parser.add_argument('--dataset_name', default='all', type=str, help='Patent data directory.')
    # parser.add_argument('--cache_dir', default='/mnt/data/HUPD/cache', type=str, help='Cache directory.')
    # parser.add_argument('--data_dir', default='"https://huggingface.co/datasets/HUPD/hupd/blob/main/hupd_metadata_jan16_2022-02-22.feather', type=str, help='Patent data directory.')
    parser.add_argument('--dataset_load_path', default='./hupd.py', type=str, help='Patent data main data load path (viz., ../patents.py).')
    parser.add_argument('--cpc_label', type=str, default=None, help='CPC label for filtering the data.')
    parser.add_argument('--ipc_label', type=str, default=None, help='IPC label for filtering the data.')
    parser.add_argument('--section', type=str, default='claims', help='Patent application section of interest.')
    parser.add_argument('--train_filing_start_date', type=str, default='', help='Start date for filtering the training data.')
    parser.add_argument('--train_filing_end_date', type=str, default='', help='End date for filtering the training data.')
    parser.add_argument('--val_filing_start_date', type=str, default="", help='Start date for filtering the training data.')
    parser.add_argument('--val_filing_end_date', type=str, default="", help='End date for filtering the validation data.')
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size (of the tokenizer).')
    parser.add_argument('--min_frequency', type=int, default=3, help='The minimum frequency that a token/word needs to have in order to appear in the vocabulary.')
    parser.add_argument('--max_length', type=int, default=512, help='The maximum total input sequence length after tokenization. Sequences longer than this number will be trunacated.')
    parser.add_argument('--use_wsampler', action='store_true', help='Use a weighted sampler (for the training set).')
    parser.add_argument('--val_set_balancer', action='store_true', help='Use a balanced set for validation? That is, do you want the same number of classes of examples in the validation set.')
    parser.add_argument('--uniform_split', default=True, help='Uniformly split the data into training and validation sets.')
    # parser.add_argument('--combine_abstract_claims', type=bool, default=True, help='Combine the abstract and claims and use that as the dataset')
    
    # Training
    parser.add_argument('--train_from_scratch', action='store_true', help='Train the model from the scratch.')
    parser.add_argument('--validation', default=False, help='Perform only validation/inference. (No performance evaluation on the training data necessary).')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size.')
    parser.add_argument('--epoch_n', type=int, default=3, help='Number of epochs (for training).')
    parser.add_argument('--val_every', type=int, default=500, help='Number of iterations we should take to perform validation.')
    parser.add_argument('--validate_training_every', type=int, default=8500, help='Number of iterations we should take to perform training validation.')
    parser.add_argument('--lr', type=float, default=2e-5, help='Model learning rate.')
    parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon value for the learning rate.')
    parser.add_argument('--wandb', action='store_true', help='Use wandb.')
    parser.add_argument('--wandb_name', type=str, default=None, help='wandb project name.')
    parser.add_argument('--pos_class_weight', type=float, default=0, help='The class weight of the rejected class label (it is 0 by default).')
    parser.add_argument('--use_scheduler', action='store_true', help='Use a scheduler.')
    parser.add_argument('--tensorboard', default=True, help='Use tensorboard.')
    parser.add_argument('--handle_skew_data', type=bool, default=True, help='Add class weights based on their fraction of the total data')

    
    # Saving purposes
    parser.add_argument('--filename', type=str, default=None, help='Name of the results file to be saved.')
    parser.add_argument('--np_filename', type=str, default=None, help='Name of the numpy file to be saved.')
    
    # Model related params
    model_path = ""
    parser.add_argument('--model_name', type=str, default="distilbert-base-uncased", help='Name of the model.')
    parser.add_argument('--embed_dim', type=int, default=200, help='Embedding dimension of the model.')
    parser.add_argument('--model_path', type=str, default=model_path, help='(Pre-trained) model path.')
    parser.add_argument('--tokenizer_path', type=str, default=model_path + "_tokenizer", help='(Pre-trained) tokenizer path.')
    parser.add_argument('--save_path', type=str, default="CS224N_models", help='The path where the model is going to be saved.')
    # parser.add_argument('--save_path', type=str, default=None, help='The path where the model is going to be saved.')

    parser.add_argument('--tokenizer_save_path', type=str, default=None, help='The path where the tokenizer is going to be saved.')
    parser.add_argument('--n_filters', type=int, default=25, help='Number of filters in the CNN (if applicable)')
    parser.add_argument('--filter_sizes', type=int, nargs='+', action='append', default=[[3,4,5], [5,6,7], [7,9,11]], help='Filter sizes for the CNN (if applicable).')
    parser.add_argument('--dropout', type=float, default=0.25, help='Use dropout for the CNN model (if applicable)')
    parser.add_argument('--naive_bayes_version', type=str, default='Bernoulli', help='Type of the Naive Bayes classifer (if applicable).')
    parser.add_argument('--alpha_smooth_val', type=float, default=1.0, help='Alpha smoothing value for the Naive Bayes classifier (if applicable).')
    
    # Parse args
    args = parser.parse_args()
    epoch_n = args.epoch_n

    # Subject area code label
    cat_label = ''
    if args.cpc_label:
        cat_label = f'CPC_{args.cpc_label}'
    elif args.ipc_label:
        cat_label = f'IPC_{args.ipc_label}'
    else:
        cat_label = 'All_IPCs'

    if args.validation and args.model_path is not None and args.tokenizer_path is None:
        args.tokenizer_path = args.model_path + '_tokenizer'

    path_params  = f"{args.section}_{args.model_name}_{args.epoch_n}_{args.batch_size}_{args.lr}_{args.max_length}_{args.embed_dim}"
    if args.save_path and not args.validation:
        now = datetime.datetime.now()
        args.save_path = f"{args.save_path}/{args.model_name}/{path_params}_date_{now.month}_{now.day}_hr_{now.hour}/"
        os.makedirs(f"{args.save_path}", exist_ok=True)
        with open(f"{args.save_path}arguments.json", "w") as file:
            json.dump(args.__dict__, file)
    
    filename = args.filename
    if filename is None:
        if args.model_name == 'naive_bayes':
            filename = f'{args.naive_bayes_version}/{cat_label}_{args.section}.txt'
        else:
            filename = f'{cat_label}_{args.section}_embdim{args.embed_dim}_maxlength{args.max_length}.txt'
    args.filename = args.save_path + filename
    write_file = open(args.filename, "w")

    tensorboard_writer = ""
    if args.tensorboard and not args.validation:
        t_path = "./tensorboard/"  + f"{path_params}_date_{now.month}_{now.day}_hr_{now.hour}"
        # os.makedirs(f"tensorboard", exist_ok=True)
        tensorboard_writer = tensorboard.SummaryWriter(log_dir=t_path)

    args.wandb_name = args.wandb_name if args.wandb_name else f'{cat_label}_{args.section}_{args.model_name}'
    
    # Make the batch size 1 when using an NB classifier
    if args.model_name == 'naive_bayes':
        args.batch_size = 1

    # Load the dataset dictionary
    dataset_dict = load_dataset(args.dataset_load_path , 
        name=args.dataset_name,
        # cache_dir=args.cache_dir,
        # data_dir=args.data_dir,
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
        dataset_dict[name] = dataset_dict[name].map(map_decision_to_string)
        # Remove the pending and CONT-patent applications
        dataset_dict[name] = dataset_dict[name].filter(lambda e: e['output'] <= 1)
    
    # Create a model and an appropriate tokenizer
    tokenizer, dataset_dict, model, vocab_size = create_model_and_tokenizer(
        args=args,
        train_from_scratch = args.train_from_scratch, 
        model_name = args.model_name, 
        dataset = dataset_dict,
        section = args.section,
        vocab_size = args.vocab_size,
        embed_dim = args.embed_dim,
        n_classes = CLASSES,
        max_length=args.max_length
        )

    print(f'*** CPC Label: {cat_label}') 
    print(f'*** Section: {args.section}')
    print(f'*** Vocabulary: {args.vocab_size}')

    if write_file:
        write_file.write(f'*** date time: {now.month}_{now.day}_hr_{now.hour}\n')
        write_file.write(f'*** CPC Label: {cat_label}\n')
        write_file.write(f'*** Section: {args.section}\n')
        write_file.write(f'*** Vocabulary: {args.vocab_size}\n')
        write_file.write(f'*** args: {args}\n\n')

    # GPU specifications 
    if args.model_name != 'naive_bayes':
        model.to(device)

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

    if not args.validation:
        # Print the statistics
        train_label_stats = dataset_statistics(args, data_loaders[0], tokenizer)
        print(f'*** Training set label statistics: {train_label_stats}')

        val_label_stats = dataset_statistics(args, data_loaders[1], tokenizer)
        print(f'*** Validation set label statistics: {val_label_stats}')

        if write_file:
            write_file.write(f'*** Training set label statistics: {train_label_stats}\n')
            write_file.write(f'*** Validation set label statistics: {val_label_stats}\n\n')
    

    if args.model_name == 'naive_bayes': 
        tokenizer.save("multilabel_ipc_nb_abstract.json") ## GET RID OF THIS
        print('Here we are!')
        train_naive_bayes(data_loaders, tokenizer, vocab_size, args.naive_bayes_version, args.alpha_smooth_val, write_file, args.np_filename)
    else:
        # Optimizer
        if args.model_name in ['logistic_regression', 'cnn', 'big_cnn', 'lstm']:
            optim = torch.optim.Adam(params=model.parameters())
        else:
            optim = torch.optim.AdamW(params=model.parameters(), lr=args.lr, eps=args.eps)
            total_steps = len(data_loaders[0]) * args.epoch_n if not args.validation else 0
        # Scheduler
        scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps = 0, num_training_steps = total_steps) if args.use_scheduler else None
        # Class weights
        if args.pos_class_weight > 0. and args.pos_class_weight < 1.:
            class_weights = torch.tensor([args.pos_class_weight, 1. - args.pos_class_weight]).to(device)
        else:
            class_weights = None

        # Loss function 
        # torch.nn.BCEWithLogitsLoss  #investigate binary loss
        # if len(CLASS_NAMES)> 2:
        if args.handle_skew_data and not args.validation:
            total_examples = sum(train_label_stats.values())
            class_weights = torch.tensor([(total_examples - train_label_stats[class_decision])/total_examples for class_decision in CLASS_NAMES]).to(device) # this should help with skewed data
        print(f"*** class weights used for loss {class_weights} class order {CLASS_NAMES}")
        write_file.write(f"*** class weights used for loss {class_weights} class order {CLASS_NAMES}")

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)  

        # wandb
        # assert wandb is not None
        if args.wandb:
            wandb_project_name = 'PatentClassification_' + cat_label
            wandb.init(project=wandb_project_name, name=args.wandb_name)
        
        if write_file:
            write_file.write(f'\nModel:\n {model}\nOptimizer: {optim}\n')
        


        # Train and validate
        if not args.validation:
            train(args, data_loaders, epoch_n, model, optim, scheduler, criterion, device, write_file, tensorboard_writer)
            if args.save_path:
                model.save_pretrained(args.save_path + "final_model")
                tokenizer.save_pretrained(args.save_path + 'final_tokenizer')
        else:
            validation(args, data_loaders[1], model, criterion, device, write_file=write_file, tensorboard_writer=tensorboard_writer)

        # Save the model
    if write_file:
        write_file.close()
