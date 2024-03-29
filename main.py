# Standard libraries and dependencies
import os
if os.getlogin() == "darke":
    PATH =  "D:/classes/cache/huggingface/hub"
    os.environ['HF_HOME'] = PATH
    os.environ['HF_DATASETS_CACHE'] = PATH

import argparse
import random
import numpy as np
from tqdm import tqdm
import datetime
import json
import pandas as pd


# PyTorch
import torch
from torch.utils import tensorboard
from torch.utils.data import DataLoader

# Hugging Face datasets
from datasets import load_dataset

# For scheduling 
from transformers import get_constant_schedule_with_warmup

#additional metrics
from torcheval.metrics import BinaryAccuracy, BinaryF1Score, BinaryAUPRC

from data_handling import map_decision_to_string, create_model_and_tokenizer, dataset_statistics, measure_accuracy, create_dataset, convert_ids_to_string

# For filtering out CONT-apps and pending apps
RANDOM_SEED = 33
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Number of classes (ACCEPTED and REJECTED)
CLASSES = 2
CLASS_NAMES = [i for i in range(CLASSES-1, -1, -1)]

# Evaluation procedure (for the neural models)
def validation(args, val_loader, model, criterion, device, name='validation', write_file=None, tensorboard_writer=None, step = 0, use_torch_metrics=True):
    if name=="train":
        val_loader = DataLoader(val_loader, batch_size=args.batch_size["validation"])
    model.eval()
    total_loss = 0.
    total_correct = 0
    total_sample = 0
    total_confusion = np.zeros((CLASSES, CLASSES))

    correct_predictions = []
    incorrect_predictions = []

    if use_torch_metrics:
        torch_metrics = {}
        torch_metrics["acc"] = BinaryAccuracy()
        torch_metrics["f1"] = BinaryF1Score()
        torch_metrics["auc"] = BinaryAUPRC()

        # Loop over the examples in the evaluation set
    for i, batch in enumerate(tqdm(val_loader)):
        # inputs, decisions, masks = batch['input_ids'], batch['labels'], batch['attention_mask']
        # inputs = inputs.to(device)
        # decisions = decisions.to(device)
        # masks = masks.to(device)
        inputs, decisions, masks, patent_numbers = batch['input_ids'], batch['labels'], batch['attention_mask'], batch['patent_number']
        inputs, decisions, masks = inputs.to(device), decisions.to(device), masks.to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=inputs, labels=decisions, attention_mask=masks)
        logits = outputs.logits
        loss = criterion(logits, decisions)
        total_loss += loss.cpu().item()

        preds = torch.argmax(logits, axis=1).flatten()
        labels = decisions.flatten()
        correct = preds == decisions
        
        for idx, was_correct in enumerate(correct):
            patent_num = patent_numbers[idx]
            if was_correct:
                correct_predictions.append(patent_num)
            else:
                incorrect_predictions.append(patent_num)
        
        # print(f"After batch {i+1}, Correct Predictions: {len(correct_predictions)}, Incorrect Predictions: {len(incorrect_predictions)}")

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
    print(f'*** F1 on the {name} set: {total_f1}')
    print(f'*** Confusion matrix:\n{total_confusion}')

    if(args.validation):
        os.makedirs(args.model_path, exist_ok=True)
        correct_predictions_path = os.path.join(args.model_path, 'correct_predictions_patent_num.json')
        incorrect_predictions_path = os.path.join(args.model_path, 'incorrect_predictions_patent_num.json')

        with open(correct_predictions_path, 'w') as f:
            json.dump(correct_predictions, f, indent=4)
        with open(incorrect_predictions_path, 'w') as f:
            json.dump(incorrect_predictions, f, indent=4)

        print(f"Total Correct Predictions: {len(correct_predictions)}")
        print(f"Total Incorrect Predictions: {len(incorrect_predictions)}")
        print("Sample Correct Predictions:", correct_predictions[:5]) 
        print("Sample Incorrect Predictions:", incorrect_predictions[:5])

    if args.tensorboard:
        tensorboard_writer.add_scalar(f'val/{name}_mean_loss', mean_loss, step)
        tensorboard_writer.add_scalar(f'val/{name}_acc', acc, step)   

        tensorboard_writer.add_scalar(f'val/{name}_torch_acc', torch_metrics["acc"].compute(), step)   
        tensorboard_writer.add_scalar(f'val/{name}_torch_f1', total_f1, step)   
        tensorboard_writer.add_scalar(f'val/{name}_torch_auc', torch_metrics["auc"].compute(), step)   


    if write_file:
        write_file.write(f'*** Accuracy on the {name} set: {total_correct/total_sample}\n')
        write_file.write(f'*** F1 on the {name} set: {total_f1}\n')
        write_file.write(f'*** Confusion matrix:\n{total_confusion}\n')
        write_file.flush()
    
    if name=="train":
        del val_loader

    return mean_loss, float(acc) * 100., total_f1


# Training procedure (for the neural models)
def train(args, data_loaders, epoch_n, model, optim, scheduler, criterion, device, write_file=None, tensorboard_writer=None):
    print('\n>>>Training starts...')
    if write_file:
        write_file.write('\n>>>Training starts...\n')
        write_file.flush()
    # Training mode is on
    model.train()
    # Best validation set accuracy so far.
    best_val_acc = 0
    best_f1 = 0
    


    for epoch in range(epoch_n):
        total_train_loss = 0.
        # Loop over the examples in the training set.
        k = 0
        loss = torch.tensor(0, dtype=torch.float32, requires_grad=True).to(device)
        for i, batch in enumerate(tqdm(data_loaders[0])):
            inputs, decisions, masks = batch['input_ids'], batch['labels'], batch['attention_mask']
            inputs = inputs.to(device, non_blocking=True)
            decisions = decisions.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            logits = model(input_ids=inputs, labels=decisions, attention_mask=masks).logits
            # logits = logits.type(torch.float32)

            # Backward pass
            if args.accumulation_steps:
                # loss = min(loss + criterion(logits, decisions),  torch.tensor(15, dtype=torch.float32, requires_grad=True).to(device))
                loss = loss + criterion(logits, decisions)  
                if k !=0 and k % args.accumulation_steps == 0:
                    total_train_loss += loss.cpu().item()
                    loss.backward()
                    if args.clip_norm:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip, error_if_nonfinite=True)
                    optim.step()
                    if scheduler:
                        scheduler.step()
                    optim.zero_grad()
                    k = 0
                    loss = torch.tensor(0, dtype=torch.float32, requires_grad=True).to(device)
                else:
                    k += 1
            else:
                loss = criterion(logits, decisions)
                print(loss)
                optim.zero_grad()
                loss.backward()
                if args.clip_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip, error_if_nonfinite=True)
                optim.step()
                if scheduler:
                    scheduler.step()
            
                total_train_loss += loss.cpu().item()

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

                if best_val_acc < val_acc:
                    best_val_acc = val_acc
                    # Save the model if a save directory is specified
                    if args.save_path:
                        # If the model is a Transformer architecture, make sure to save the tokenizer as well
                        if args.model_name in ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base', 'gpt2', 'allenai/longformer-base-4096', 'mistralai/Mistral-7B-v0.1']:
                            model.save_pretrained(args.save_path + 'model')
                            tokenizer.save_pretrained(args.save_path + 'tokenizer')
                        else:
                            torch.save(model.state_dict(), args.save_path)
    
        if args.save_path:
            # If the model is a Transformer architecture, make sure to save the tokenizer as well
            if args.model_name in ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base', 'gpt2', 'allenai/longformer-base-4096', 'mistralai/Mistral-7B-v0.1']:
                model.save_pretrained(args.save_path + 'epoch_model')
                tokenizer.save_pretrained(args.save_path + 'epoch_tokenizer')
            else:
                torch.save(model.state_dict(), args.save_path) 
        # if (epoch * len(data_loaders[0]) + i) % args.validate_training_every_epoch == 0 and i !=0:
        #     validation(args, data_loaders[0], model, criterion, device, name='train', tensorboard_writer=tensorboard_writer, step=epoch * len(data_loaders[0]) + i)

    # Training is complete!
    print(f'\n ~ The End ~')
    if write_file:
        write_file.write('\n ~ The End ~\n')
    
    # Final evaluation on the validation set
    _, val_acc, f1_acc = validation(args, data_loaders[1], model, criterion, device, name='validation', write_file=write_file, tensorboard_writer=tensorboard_writer, step=epoch * len(data_loaders[0]) + i)
    best_f1 = max(best_f1, f1_acc)
    if best_val_acc < val_acc:
        best_val_acc = val_acc
        
        # Save the best model so far
        if args.save_path:
            if args.model_name in ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base', 'gpt2', 'allenai/longformer-base-4096', 'mistralai/Mistral-7B-v0.1']:
                model.save_pretrained(args.save_path + 'model')
            else:
                torch.save(model.state_dict(), args.save_path)

    
    # Print the highest accuracy score obtained by the model on the validation set
    print(f'*** Highest accuracy on the validation set: {best_val_acc}.')
    print(f'*** Highest f1 accuract on the validation set: {best_f1}.')

    if write_file:
        write_file.write(f'\n*** Highest accuracy on the validation set: {best_val_acc}.')
        write_file.write(f'\n*** Highest f1 accuracy on the validation set: {best_f1}.')
    

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    parser = argparse.ArgumentParser()
    
    # Dataset
    parser.add_argument('--dataset_name', default='all', type=str, help='Patent data directory.')
    parser.add_argument('--dataset_load_path', default='./hupd.py', type=str, help='Patent data main data load path (viz., ../patents.py).')
    parser.add_argument('--cpc_label', type=str, default=None, help='CPC label for filtering the data.')
    parser.add_argument('--ipc_label', type=str, default=None, help='IPC label for filtering the data.')
    parser.add_argument('--section', type=str, default='claims', help='Patent application section of interest.')
    parser.add_argument('--train_filing_start_date', type=str, default='2008-01-01', help='Start date for filtering the training data.')
    parser.add_argument('--train_filing_end_date', type=str, default='2012-01-15', help='End date for filtering the training data.')
    parser.add_argument('--val_filing_start_date', type=str, default='2008-01-01', help='Start date for filtering the training data.')
    parser.add_argument('--val_filing_end_date', type=str, default='2014-12-31', help='End date for filtering the validation data.')
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size (of the tokenizer).')
    parser.add_argument('--use_wsampler', action='store_true', help='Use a weighted sampler (for the training set).')
    parser.add_argument('--val_set_balancer', action='store_true', help='Use a balanced set for validation? That is, do you want the same number of classes of examples in the validation set.')
    parser.add_argument('--uniform_split', default=True, help='Uniformly split the data into training and validation sets.')
    parser.add_argument('--num_proc', default=8, help='Number of processors to use for preprocessing data')
    # parser.add_argument('--combine_abstract_claims', type=bool, default=True, help='Combine the abstract and claims and use that as the dataset')
    
    # Training
    parser.add_argument('--accumulation_steps', default=8, help='Num steps to accum gradient')
    parser.add_argument('--train_from_scratch', action='store_true', help='Train the model from the scratch.')
    parser.add_argument('--validation', default=True, help='Perform only validation/inference. (No performance evaluation on the training data necessary).')
    # parser.add_argument('--validation', action='store_true', help='Enable validation mode')

    parser.add_argument('--batch_size', type=dict, default={'train':1, 'validation':1}, help='Batch size.')
    parser.add_argument('--epoch_n', type=int, default=1, help='Number of epochs (for training).')
    parser.add_argument('--val_every', type=int, default=2100, help='Number of iterations we should take to perform validation.')
    parser.add_argument('--validate_training_every', type=int, default=8500, help='Number of iterations we should take to perform training validation.')
    parser.add_argument('--lr', type=float, default=3e-4, help='Model learning rate.')
    parser.add_argument('--wandb', action='store_true', help='Use wandb.')
    parser.add_argument('--wandb_name', type=str, default=None, help='wandb project name.')
    parser.add_argument('--use_scheduler', default=True, help='Use a scheduler.')
    parser.add_argument('--tensorboard', default=True, help='Use tensorboard.')
    parser.add_argument('--handle_skew_data', type=bool, default=True, help='Add class weights based on their fraction of the total data')
    parser.add_argument('--continue_training', type=bool, default=False, help='Load weights and continue training')
    parser.add_argument('--linear_probe', type=bool, default=False, help='Load weights and continue training')
    parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon value for the learning rate.')
    parser.add_argument('--warmup_ratio', type=float, default=0.03, help='Fraction of steps to do a warmup for')
    parser.add_argument('--grad_norm_clip', type=float, default=1, help='Clip the grad norm')
    parser.add_argument('--clip_norm', type=bool, default=True , help='Clip the grad norm')
    parser.add_argument('--use_flash_attention_2', type=bool, default=False, help='Use flash attention')
    parser.add_argument('--QloRA', type=bool, default=True, help='Use QloRA')
    parser.add_argument('--weight_decay', default=0.01, help='Use weight_decay')



    # Saving purposes
    parser.add_argument('--filename', type=str, default=None, help='Name of the results file to be saved.')
    

    mistral_model_name = "mistralai/Mistral-7B-v0.1"
    # Model related params
    # model_path = "CS224N_models/mistralai/Mistral-7B-v0.1/claims_Mistral-7B-v0.1_2_8_0.0001_512_False_sample_False_date_3_3_hr_18/epoch_"
    model_path = "CS224N_models/mistralai/Mistral-7B-v0.1/claims_Mistral-7B-v0.1_1_8_0.0003_512_False_all_False_date_3_13_hr_7/"
    parser.add_argument('--model_name', type=str, default=mistral_model_name, help='Name of the model.')
    parser.add_argument('--model_path', type=str, default=model_path + "model", help='(Pre-trained) model path.')
    parser.add_argument('--tokenizer_path', type=str, default=model_path + "tokenizer", help='(Pre-trained) tokenizer path.')
    parser.add_argument('--save_path', type=str, default="CS224N_models", help='The path where the model is going to be saved.')
    # parser.add_argument('--save_path', type=str, default=None, help='The path where the model is going to be saved.')

    parser.add_argument('--tokenizer_save_path', type=str, default=None, help='The path where the tokenizer is going to be saved.')
    parser.add_argument('--max_length', type=int, default=512, help='The maximum total input sequence length after tokenization. Sequences longer than this number will be trunacated.')

    
    # Parse args
    args = parser.parse_args()
    epoch_n = args.epoch_n
    args.device = device

    # Subject area code label
    cat_label = ''
    if args.cpc_label:
        cat_label = f'CPC_{args.cpc_label}'
    elif args.ipc_label:
        cat_label = f'IPC_{args.ipc_label}'
    else:
        cat_label = 'All_IPCs'


    path_params  = f"{args.section}_{args.model_name.split('/')[-1]}_{args.epoch_n}_{args.batch_size['train'] if not args.accumulation_steps else args.accumulation_steps*args.batch_size['train']}_{args.lr}_{args.max_length}_{args.continue_training}_{args.dataset_name}_{args.linear_probe}"

    if args.save_path and not args.validation:
        now = datetime.datetime.now()
        args.save_path = f"{args.save_path}/{args.model_name}/{path_params}_date_{now.month}_{now.day}_hr_{now.hour}/"
        os.makedirs(f"{args.save_path}", exist_ok=True)
        with open(f"{args.save_path}arguments.json", "w") as file:
            json.dump(args.__dict__, file)
    
    filename = args.filename
    if filename is None:
        filename = f'{cat_label}_{args.section}_maxlength{args.max_length}.txt'
    args.filename = args.save_path + filename

    if args.validation:
        write_file = ""
        args.dataset_name = "all"
        args.tensorboard = None
        # args.uniform_split = False
        # args.val_set_balancer = True
        args.train_filing_start_date = '2015-01-01'
        args.train_filing_end_date = '2015-01-15'
        args.val_filing_start_date = '2015-01-16'
        args.val_filing_end_date = '2015-2-15'
        args.uniform_split = True
        args.val_set_balancer = True
    else:
        write_file = open(args.filename, "w")

    tensorboard_writer = ""
    if args.tensorboard and not args.validation:
        t_path = "./tensorboard/"  + f"{path_params}_date_{now.month}_{now.day}_hr_{now.hour}"
        # os.makedirs(f"tensorboard", exist_ok=True)
        tensorboard_writer = tensorboard.SummaryWriter(log_dir=t_path)

    args.wandb_name = args.wandb_name if args.wandb_name else f'{cat_label}_{args.section}_{args.model_name}'

    # print(f"train start date: {args.train_filing_start_date}\ntrain end date: {args.train_filing_end_date}\nval start data: {args.val_filing_start_date}\nval end date: {args.val_filing_end_date}")
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
        dataset_dict[name] = dataset_dict[name].filter(lambda e: e['labels'] <= 1)
        dataset_dict[name] = dataset_dict[name].remove_columns(set(dataset_dict[name].column_names) - set(["labels", args.section]))

    
    # Create a model and an appropriate tokenizer
    tokenizer, dataset_dict, model = create_model_and_tokenizer(
        args=args,
        train_from_scratch = args.train_from_scratch, 
        model_name = args.model_name, 
        dataset = dataset_dict,
        max_length=args.max_length
        )

    print(f'*** CPC Label: {cat_label}') 
    print(f'*** Section: {args.section}')
    print(f'*** Vocabulary: {args.vocab_size}')

    if write_file:
        write_file.write(f'*** date time: {now.month}_{now.day}_hr_{now.hour}\n')
        write_file.write(f'*** CPC Label: {cat_label}\n')
        write_file.write(f'*** Section: {args.section}\n')
        write_file.write(f'*** args: {args}\n\n')


    # Load the dataset
    data_loaders = create_dataset(
        args = args, 
        dataset_dict = dataset_dict, 
        tokenizer = tokenizer, 
        section = args.section,
        )
    del dataset_dict

    if not args.validation:
        # Print the statistics
        trainable_params = model.print_trainable_parameters()
        write_file.write(f"{trainable_params}\n")
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

    # optim = torch.optim.AdamW(params=model.parameters() if not args.linear_probe else params, lr=args.lr, eps=args.eps)
    optim = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas = (0.9, 0.95))


    # Instantiate scheduler
    if not args.validation:
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optim,
            num_warmup_steps=5,
        ) if args.use_scheduler else None


    class_weights = None
    if args.handle_skew_data and not args.validation:
        total_examples = sum(train_label_stats.values())
        class_weights = torch.tensor([(train_label_stats[class_decision])/total_examples for class_decision in CLASS_NAMES]).to(device) # this should help with skewed data.  
        # The weights are in order 0:weight, 1:weight.  Since the class names are in descending order, we have the weight be the fraction instead of the total - fraction.

    print(f"*** class weights used for loss {class_weights} class order {CLASS_NAMES}")
    if write_file:
        write_file.write(f"*** class weights used for loss {class_weights} class order {CLASS_NAMES}")

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)  

    if write_file:
        write_file.write(f'\nModel:\n {model}\nOptimizer: {optim}\n')
    
    # Train and validate
    if not args.validation:
        train(args, data_loaders, epoch_n, model, optim, scheduler, criterion, device, write_file, tensorboard_writer)
    else:
        validation(args, data_loaders[1], model, criterion, device, write_file=write_file, tensorboard_writer=tensorboard_writer)

        # Save the model
    if write_file:
        write_file.close()
