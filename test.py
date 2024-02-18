import os
# PATH =  "D:/classes/cache/huggingface/hub"
# os.environ['TRANSFORMERS_CACHE'] = PATH
# os.environ['HF_HOME'] = PATH
# os.environ['HF_DATASETS_CACHE'] = PATH
# os.environ['TORCH_HOME'] = PATH
from datasets import load_dataset

dataset_dict = load_dataset('HUPD/hupd',
    name='sample',
    cache_dir="D:/classes/CS224N/HUPD_dataset/data/",
    data_files="https://huggingface.co/datasets/HUPD/hupd/blob/main/hupd_metadata_jan16_2022-02-22.feather", 
    data_dir="D:/classes/CS224N/HUPD_dataset/data/",
    icpr_label=None,
    train_filing_start_date='2016-01-01',
    train_filing_end_date='2016-01-21',
    val_filing_start_date='2016-01-22',
    val_filing_end_date='2016-01-31',
)   

print(dataset_dict)