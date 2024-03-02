# Standard libraries and dependencies
import os
if os.getlogin() == "darke":
    PATH =  "D:/classes/cache/huggingface/hub"
    os.environ['HF_HOME'] = PATH
    os.environ['HF_DATASETS_CACHE'] = PATH

import argparse
import random
import numpy as np
import datetime
import json
# import lora, mistal7b

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

# For scheduling 
from transformers import get_linear_schedule_with_warmup


from data_handling import map_decision_to_string, create_model_and_tokenizer, dataset_statistics, create_dataset

# For filtering out CONT-apps and pending apps
RANDOM_SEED = 1729
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Number of classes (ACCEPTED and REJECTED)
CLASSES = 2
CLASS_NAMES = [i for i in range(CLASSES-1, -1, -1)]

def main():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--dataset_name', default='all', type=str, help='Patent data directory.')
    parser.add_argument('--dataset_load_path', default='./hupd.py', type=str, help='Patent data main data load path (viz., ../patents.py).')
    parser.add_argument('--cpc_label', type=str, default=None, help='CPC label for filtering the data.')
    parser.add_argument('--ipc_label', type=str, default=None, help='IPC label for filtering the data.')
    parser.add_argument('--section', type=str, default='claims', help='Patent application section of interest.')
    parser.add_argument('--train_filing_start_date', type=str, default='', help='Start date for filtering the training data.')
    parser.add_argument('--train_filing_end_date', type=str, default='', help='End date for filtering the training data.')
    parser.add_argument('--val_filing_start_date', type=str, default='', help='Start date for filtering the training data.')
    parser.add_argument('--val_filing_end_date', type=str, default='', help='End date for filtering the validation data.')
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size (of the tokenizer).')
    parser.add_argument('--min_frequency', type=int, default=3, help='The minimum frequency that a token/word needs to have in order to appear in the vocabulary.')
    parser.add_argument('--use_wsampler', action='store_true', help='Use a weighted sampler (for the training set).')
    parser.add_argument('--val_set_balancer', action='store_true', help='Use a balanced set for validation? That is, do you want the same number of classes of examples in the validation set.')
    parser.add_argument('--uniform_split', default=True, help='Uniformly split the data into training and validation sets.')
    parser.add_argument('--num_proc', default=8, help='Number of processors to use for preprocessing data')
    # parser.add_argument('--combine_abstract_claims', type=bool, default=True, help='Combine the abstract and claims and use that as the dataset')
    
    # Training
    parser.add_argument('--accumulation_steps', default=0, help='Num steps to accum gradient')
    parser.add_argument('--train_from_scratch', action='store_true', help='Train the model from the scratch.')
    parser.add_argument('--validation', default=False, help='Perform only validation/inference. (No performance evaluation on the training data necessary).')
    parser.add_argument('--batch_size', type=dict, default={'train':8, 'validation':48}, help='Batch size.')
    parser.add_argument('--epoch_n', type=int, default=2, help='Number of epochs (for training).')
    parser.add_argument('--val_every', type=int, default=2000, help='Number of iterations we should take to perform validation.')
    parser.add_argument('--validate_training_every', type=int, default=8500, help='Number of iterations we should take to perform training validation.')
    parser.add_argument('--lr', type=float, default=2e-5, help='Model learning rate.')
    parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon value for the learning rate.')
    parser.add_argument('--wandb', action='store_true', help='Use wandb.')
    parser.add_argument('--wandb_name', type=str, default=None, help='wandb project name.')
    parser.add_argument('--pos_class_weight', type=float, default=0, help='The class weight of the rejected class label (it is 0 by default).')
    parser.add_argument('--use_scheduler', action='store_true', help='Use a scheduler.')
    parser.add_argument('--tensorboard', default=True, help='Use tensorboard.')
    parser.add_argument('--handle_skew_data', type=bool, default=True, help='Add class weights based on their fraction of the total data')
    parser.add_argument('--continue_training', type=bool, default=True, help='Load weights and continue training')
    parser.add_argument('--linear_probe', type=bool, default=False, help='Load weights and continue training')


    
    # Saving purposes
    parser.add_argument('--filename', type=str, default=None, help='Name of the results file to be saved.')
    parser.add_argument('--np_filename', type=str, default=None, help='Name of the numpy file to be saved.')
    

    mistral_model_name = "distilbert-base-uncased"
    # Model related params
    model_path = "CS224N_models/distilbert-base-uncased/claims_distilbert-base-uncased_2_8_2e-05_512_200_False_all_True_date_3_1_hr_21/f1_"
    parser.add_argument('--model_name', type=str, default=mistral_model_name, help='Name of the model.')
    parser.add_argument('--embed_dim', type=int, default=200, help='Embedding dimension of the model.')
    parser.add_argument('--model_path', type=str, default=model_path + "model", help='(Pre-trained) model path.')
    parser.add_argument('--tokenizer_path', type=str, default=model_path + "tokenizer", help='(Pre-trained) tokenizer path.')
    parser.add_argument('--save_path', type=str, default="CS224N_models", help='The path where the model is going to be saved.')
    # parser.add_argument('--save_path', type=str, default=None, help='The path where the model is going to be saved.')

    parser.add_argument('--tokenizer_save_path', type=str, default=None, help='The path where the tokenizer is going to be saved.')
    parser.add_argument('--n_filters', type=int, default=25, help='Number of filters in the CNN (if applicable)')
    parser.add_argument('--filter_sizes', type=int, nargs='+', action='append', default=[[3,4,5], [5,6,7], [7,9,11]], help='Filter sizes for the CNN (if applicable).')
    parser.add_argument('--dropout', type=float, default=0.25, help='Use dropout for the CNN model (if applicable)')
    parser.add_argument('--naive_bayes_version', type=str, default='Bernoulli', help='Type of the Naive Bayes classifer (if applicable).')
    parser.add_argument('--alpha_smooth_val', type=float, default=1.0, help='Alpha smoothing value for the Naive Bayes classifier (if applicable).')

    parser.add_argument('--max_length', type=int, default=512, help='The maximum total input sequence length after tokenization. Sequences longer than this number will be trunacated.')

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


    path_params  = f"{args.section}_{args.model_name}_{args.epoch_n}_{args.batch_size['train']}_{args.lr}_{args.max_length}_{args.embed_dim}_{args.continue_training}_{args.dataset_name}_{args.linear_probe}"
    if args.save_path and not args.validation:
        now = datetime.datetime.now()
        args.save_path = f"{args.save_path}/{args.model_name}/{path_params}_date_{now.month}_{now.day}_hr_{now.hour}/"
        os.makedirs(f"{args.save_path}", exist_ok=True)
        with open(f"{args.save_path}arguments.json", "w") as file:
            json.dump(args.__dict__, file)
    
    filename = args.filename
    if filename is None:
        filename = f'{cat_label}_{args.section}_embdim{args.embed_dim}_maxlength{args.max_length}.txt'
    args.filename = args.save_path + filename

    if args.validation:
        write_file = ""
        args.dataset_name = "sample"
        args.tensorboard = None
        args.uniform_split = False
        args.val_set_balancer = True
        args.train_filing_start_date = '2016-01-01'
        args.train_filing_end_date = '2016-01-21'
        args.val_filing_start_date = '2016-01-01'
        args.val_filing_end_date = '2016-01-31'

    else:
        write_file = open(args.filename, "w")

    tensorboard_writer = ""
    if args.tensorboard and not args.validation:
        t_path = "./tensorboard/"  + f"{path_params}_date_{now.month}_{now.day}_hr_{now.hour}"
        # os.makedirs(f"tensorboard", exist_ok=True)
        tensorboard_writer = tensorboard.SummaryWriter(log_dir=t_path)

    args.wandb_name = args.wandb_name if args.wandb_name else f'{cat_label}_{args.section}_{args.model_name}'
    
    # Make the batch size 1 when using an NB classifier

    # Load the dataset dictionary
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
        train_label_stats = dataset_statistics( data_loaders[0])
        print(f'*** Training set label statistics: {train_label_stats}')


    val_label_stats = dataset_statistics( data_loaders[1])
    print(f'*** Validation set label statistics: {val_label_stats}')
    # print(f'*** Training set longest {args.section}: {longest_train_section}')
    # print(f'*** Training set longest {args.section}: {longest_val_section}')
    # write_file.write(f'*** Training set longest {args.section}: {longest_train_section}')
    # write_file.write(f'*** Training set longest {args.section}: {longest_val_section}')



    if write_file:
        write_file.write(f'*** Training set label statistics: {train_label_stats}\n')
        write_file.write(f'*** Validation set label statistics: {val_label_stats}\n\n')

    if args.linear_probe:
        params = list(model.get_submodule("pre_classifier").parameters()) + list(model.get_submodule("classifier").parameters())

    optim = torch.optim.AdamW(params=model.parameters() if not args.linear_probe else params, lr=args.lr, eps=args.eps)
    total_steps = len(data_loaders[0]) * args.epoch_n if not args.validation else 0
    # Scheduler
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps = 0, num_training_steps = total_steps) if args.use_scheduler else None

    # Loss function 
    # torch.nn.BCEWithLogitsLoss  #investigate binary loss
    # if len(CLASS_NAMES)> 2:
    if args.handle_skew_data and not args.validation:
        total_examples = sum(train_label_stats.values())
        class_weights = torch.tensor([(train_label_stats[class_decision])/total_examples for class_decision in CLASS_NAMES]).to(device) # this should help with skewed data.  
        # The weights are in order 0:weight, 1:weight.  Since the class names are in descending order, we have the weight be the fraction instead of the total - fraction.

    print(f"*** class weights used for loss {class_weights} class order {CLASS_NAMES}")
    if write_file:
        write_file.write(f"*** class weights used for loss {class_weights} class order {CLASS_NAMES}")

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)  


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
