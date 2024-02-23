import os
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import multiprocess as mp
from multiprocesspandas import applyparallel

from parl import paraleliz2

try:
    import ujson as json
except:
    import json

import datasets



def main():
    ipcr_label = None
    cpc_label = None
    train_filing_start_date = None
    train_filing_end_date = None
    val_filing_start_date = None
    val_filing_end_date = None
    query_string = None
    val_set_balancer = None
    uniform_split = True
    force_extract = None
    processes = 8
    name="all"
    description="Patent data from all years (2004-2018)"
    metadata_url="D:/classes/CS224N/HUPD_dataset/hupd_metadata_2022-02-22.feather"
    data_url=""
    data_dir="D:/classes/CS224N/HUPD_dataset/data"
    RANDOM_STATE = 1729


        # Download metadata
    # NOTE: Metadata is stored as a Pandas DataFrame in Apache Feather format

    metadata_file = metadata_url

    # Download data
    # NOTE: The extracted path contains a subfolder, data_dir. This directory holds
    # a large number of json files (one json file per patent application).

    json_dir = os.path.join("", data_dir)

    # Load metadata file
    print(f'Reading metadata file: {metadata_file}')
    if metadata_url.endswith('.feather'):
        df = pd.read_feather(metadata_file)
    elif metadata_url.endswith('.csv'):
        df = pd.read_csv(metadata_file)
    elif metadata_url.endswith('.tsv'):
        df = pd.read_csv(metadata_file, delimiter='\t')
    elif metadata_url.endswith('.pickle'):
        df = pd.read_pickle(metadata_file)
    else:
        raise ValueError(f'Metadata file invalid: {metadata_url}')

    # Filter based on ICPR / CPC label
    if ipcr_label:
        print(f'Filtering by IPCR label: {ipcr_label}')
        df = df[df['main_ipcr_label'].str.startswith(ipcr_label)]
    elif cpc_label:
        print(f'Filtering by CPC label: {cpc_label}')
        df = df[df['main_cpc_label'].str.startswith(cpc_label)]

    # Filter metadata based on arbitrary query string
    if query_string:
        df = df.query(query_string)
        
    if force_extract:
        if name == 'all':
            if train_filing_start_date and val_filing_end_date:
                if train_filing_end_date and val_filing_start_date:
                    training_year_range = set(range(int(train_filing_start_date[:4]), int(train_filing_end_date[:4]) + 1))
                    validation_year_range = set(range(int(val_filing_start_date[:4]), int(val_filing_end_date[:4]) + 1))
                    full_year_range = training_year_range.union(validation_year_range)
                else:
                    full_year_range = set(range(int(train_filing_start_date[:4]), int(val_filing_end_date[:4]) + 1))
            else:
                full_year_range = set(range(2004, 2019))
            

            import tarfile
            print("local ", "*" * 20)
            for year in full_year_range:
                tar_file_path = f'{json_dir}/{year}.tar.gz'
                print(f'Extracting {tar_file_path}')
                # open file
                tar_file = tarfile.open(tar_file_path)
                # extracting file
                tar_file.extractall(f'{json_dir}')
                tar_file.close()    

    if name == 'all':
        final_df = pd.DataFrame()
        for file in os.listdir(json_dir):
            year = file.split(".")
            if len(year) == 1:
                print(year)
                start  = f"{year[0]}-01-01"
                end = f"{year[0]}-12-31"
                start_df = df[df['filing_date'] >= start]
                final_df = pd.concat([final_df, start_df[start_df['filing_date'] <= end]])
        df = final_df
        del final_df
        del start_df       


    # Train-validation split (either uniform or by date)
    if uniform_split:

        # Assumes that training_start_data < val_end_date
        if train_filing_start_date:
            df = df[df['filing_date'] >= train_filing_start_date]
        if val_filing_end_date:
            df = df[df['filing_date'] <= val_filing_end_date]
        df = df.sample(frac=1.0, random_state=RANDOM_STATE)
        num_train_samples = int(len(df) * 0.85)
        train_df = df.iloc[0:num_train_samples]
        val_df = df.iloc[num_train_samples:-1]

    else:

        # Check
        if not (train_filing_start_date and train_filing_end_date and
                val_filing_start_date and train_filing_end_date):
            raise ValueError("Please either use uniform_split or specify your exact \
                training and validation split dates.")

        # Does not assume that training_start_data < val_end_date
        print(f'Filtering train dataset by filing start date: {train_filing_start_date}')
        print(f'Filtering train dataset by filing end date: {train_filing_end_date}')
        print(f'Filtering val dataset by filing start date: {val_filing_start_date}')
        print(f'Filtering val dataset by filing end date: {val_filing_end_date}')
        train_df = df[
            (df['filing_date'] >= train_filing_start_date) & 
            (df['filing_date'] < train_filing_end_date)
        ]
        val_df = df[
            (df['filing_date'] >= val_filing_start_date) & 
            (df['filing_date'] < val_filing_end_date)
        ]

    # TODO: We can probably make this step faster
    if val_set_balancer:
        rejected_df = val_df[val_df.decision == 'REJECTED']
        num_rejected = len(rejected_df)
        accepted_df = val_df[val_df.decision == 'ACCEPTED']
        num_accepted = len(accepted_df)
        if num_rejected < num_accepted:
            accepted_df = accepted_df.sample(frac=1.0, random_state=RANDOM_STATE)  # shuffle(accepted_df)
            accepted_df = accepted_df[:num_rejected]
        else:
            rejected_df = rejected_df.sample(frac=1.0, random_state=RANDOM_STATE)  # shuffle(rejected_df)
            rejected_df = rejected_df[:num_accepted]
        val_df = pd.concat([rejected_df, accepted_df])



    return val_df,json_dir


    # a = datasets.Dataset.from_pandas(train_df)
    # v = datasets.Dataset.from_pandas(val_df)

    # partial_train = pd.DataFrame()
    # partial_val = pd.DataFrame()


    # def multiprocess_train(data):
    #     datasets.SplitGenerator(
    #         name=datasets.Split.TRAIN,
    #         gen_kwargs=dict(  # these kwargs are passed to _generate_examples
    #         df=data,
    #         json_dir=json_dir,
    #         split='train',
    #         num_procs=processes
    #     ))

    # def multiprocess_val(data):
    #     datasets.SplitGenerator(
    #         name=datasets.Split.VALIDATION,
    #         gen_kwargs=dict(
    #             df=data,
    #             json_dir=json_dir,
    #             split='val',
    #             num_procs=processes
    #         )
    #     )
    # # print(val_df)
    # # val_df = val_df.map(multiprocess_val, num_proc=processes)
    # # # train_df.map(multiprocess_train, num_proc=processes)
    # # print(val_df)
    #         # chunk_size = int(dataset.shape[0] / num_procs)
    # # with ThreadPoolExecutor(num_workers = num_procs) as pool:
    # #     for start in range(0, dataset.shape[0], chunk_size):
    # #         yield pool.map(inner, dataset.iloc[start:start + chunk_size])
        

    # partial_train = pd.DataFrame()
    # partial_val = pd.DataFrame()
    # print(val_df)
    
    # a.map(lambda e: paraleliz2(e), num_proc=2, batched=True)
    # # partial_train.map(paraleliz2, num_proc=processes))
    # # train_df.map(multiprocess_train, num_proc=processes)
    # print(val_df)
if __name__ == "__main__":
    val_df, json_dir = main()
    chunk_size = int(val_df.shape[0] / (mp.cpu_count() - 1))
    final = pd.DataFrame()
    data = datasets.Dataset.from_pandas(val_df)
    data  = data.map(paraleliz2, num_proc=4)
    # with mp.Pool(1) as pool:
    #     # for start in range(0, val_df.shape[0], chunk_size):
    #     #     dataset = val_df.iloc[start:start + chunk_size]
    #     for id_, x in enumerate(val_df.itertuples()):
    #         # args = (id_, x)
    #         res=pool.map(paraleliz2, x.copy())
    print(data["comb"])