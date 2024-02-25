"""
The Harvard USPTO Patent Dataset (HUPD) is a large-scale, well-structured, and multi-purpose corpus 
of English-language patent applications filed to the United States Patent and Trademark Office (USPTO) 
between 2004 and 2018. With more than 4.5 million patent documents, HUPD is two to three times larger 
than comparable corpora. Unlike other NLP patent datasets, HUPD contains the inventor-submitted versions 
of patent applications, not the final versions of granted patents, allowing us to study patentability at 
the time of filing using NLP methods for the first time.
"""

from __future__ import absolute_import, division, print_function

import os
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
try:
    import ujson as json
except:
    import json

import datasets


_CITATION = """\
@InProceedings{suzgun2021:hupd,
title = {The Harvard USPTO Patent Dataset},
authors={Mirac Suzgun and Suproteem Sarkar and Luke Melas-Kyriazi and Scott Kominers and Stuart Shieber},
year={2021}
}
"""

_DESCRIPTION = """
The Harvard USPTO Patent Dataset (HUPD) is a large-scale, well-structured, and multi-purpose corpus 
of English-language patent applications filed to the United States Patent and Trademark Office (USPTO) 
between 2004 and 2018. With more than 4.5 million patent documents, HUPD is two to three times larger 
than comparable corpora. Unlike other NLP patent datasets, HUPD contains the inventor-submitted versions 
of patent applications, not the final versions of granted patents, allowing us to study patentability at 
the time of filing using NLP methods for the first time.
"""

RANDOM_STATE = 1729

_FEATURES = [
    "patent_number",
    "decision",
    "title",
    "abstract",
    "claims",
    "background",
    "summary",
    "description",
    "cpc_label",
    "ipc_label",
    "filing_date",
    "patent_issue_date",
    "date_published",
    "examiner_id"
]


def str_to_date(s):
    """A helper function to convert strings to dates"""
    return datetime.datetime.strptime(s, '%Y-%m-%d')


class PatentsConfig(datasets.BuilderConfig):
    """BuilderConfig for Patents"""

    def __init__(
        self,
        metadata_url: str,
        data_url: str,
        data_dir: str,
        ipcr_label: str = None,
        cpc_label: str = None,
        train_filing_start_date: str = None,
        train_filing_end_date: str = None,
        val_filing_start_date: str = None,
        val_filing_end_date: str = None,
        query_string: str = None,
        val_set_balancer=False,
        uniform_split=False,
        force_extract=False,
        **kwargs
    ):
        """
        If train_filing_end_date is None, then a random train-val split will be used. If it is 
        specified, then the specified date range will be used for the split. If train_filing_end_date 
        if specified and val_filing_start_date is not specifed, then val_filing_start_date defaults to 
        train_filing_end_date. 

        Args:
            metadata_url: `string`, url from which to download the metadata file
            data_url: `string`, url from which to download the json files
            data_dir: `string`, folder (in cache) in which downloaded json files are stored
            ipcr_label: International Patent Classification code
            cpc_label: Cooperative Patent Classification code
            train_filing_start_date: Start date for patents in train set (and val set if random split is used)
            train_filing_end_date: End date for patents in train set
            val_filing_start_date: Start date for patents in val set
            val_filing_end_date: End date for patents in val set (and train set if random split is used)
            force_extract: Extract only the relevant years if this parameter is used.
            **kwargs: keyword arguments forwarded to super
        """
        super().__init__(**kwargs)
        self.metadata_url = metadata_url
        self.data_url = data_url
        self.data_dir = data_dir
        self.ipcr_label = ipcr_label
        self.cpc_label = cpc_label
        self.train_filing_start_date = train_filing_start_date
        self.train_filing_end_date = train_filing_end_date
        self.val_filing_start_date = val_filing_start_date
        self.val_filing_end_date = val_filing_end_date
        self.query_string = query_string
        self.val_set_balancer = val_set_balancer
        self.uniform_split = uniform_split
        self.force_extract = force_extract



class Patents(datasets.GeneratorBasedBuilder):
    _DESCRIPTION

    VERSION = datasets.Version("1.0.2")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    BUILDER_CONFIG_CLASS = PatentsConfig
    if os.getlogin() == "darke":
        BUILDER_CONFIGS = [
            PatentsConfig(
                name="sample", 
                description="Patent data from January 2016, for debugging", 
                metadata_url="D:/classes/CS224N/HUPD_dataset/sample_hupd_metadata_jan16_2022-02-22.feather",
                data_url="",
                data_dir="D:/classes/CS224N/HUPD_dataset/sample/sample",   # this will unpack to data/{year}
            ),
            PatentsConfig(
                name="all", 
                description="Patent data from all years (2004-2018)", 
                metadata_url="D:/classes/CS224N/HUPD_dataset/hupd_metadata_2022-02-22.feather",
                data_url="",
                data_dir="D:/classes/CS224N/HUPD_dataset/data",   # this will unpack to data/{year}
            ),
            
        ]
    else:
        BUILDER_CONFIGS = [
            PatentsConfig(
                name="sample", 
                description="Patent data from January 2016, for debugging", 
                metadata_url="./hupd_metadata_jan16_2022-02-22.feather",
                data_url="",
                data_dir="./data/sample",   # this will unpack to data/{year}
            ),
            PatentsConfig(
                name="all", 
                description="Patent data from all years (2004-2018)", 
                metadata_url="./hupd_metadata_2022-02-22.feather",
                data_url="",
                data_dir="./data",   # this will unpack to data/{year}
            ),
            
        ]


    # BUILDER_CONFIGS = [
    #     PatentsConfig(
    #         name="sample", 
    #         description="Patent data from January 2016, for debugging", 
    #         metadata_url="https://huggingface.co/datasets/HUPD/hupd/resolve/main/hupd_metadata_jan16_2022-02-22.feather",
    #         data_url="https://huggingface.co/datasets/HUPD/hupd/resolve/main/data/sample-jan-2016.tar.gz",
    #         data_dir="sample",  # this will unpack to data/sample/2016
    #     ),
    #     PatentsConfig(
    #         name="all", 
    #         description="Patent data from all years (2004-2018)", 
    #         metadata_url="https://huggingface.co/datasets/HUPD/hupd/resolve/main/hupd_metadata_2022-02-22.feather",
    #         data_url="https://huggingface.co/datasets/HUPD/hupd/resolve/main/data/all-years.tar",
    #         data_dir="data",   # this will unpack to data/{year}
    #     ),
    # ]


    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datasets.Features(
                {k: datasets.Value("string") for k in _FEATURES}
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=("claims", "decision"),
            homepage="https://github.com/suzgunmirac/hupd",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        """Returns SplitGenerators."""
        print(f'Loading dataset with config: {self.config}')

        # Download metadata
        # NOTE: Metadata is stored as a Pandas DataFrame in Apache Feather format
        metadata_url = self.config.metadata_url
        metadata_file = metadata_url
        if self.config.metadata_url[0:5] == "https":
            metadata_file = dl_manager.download_and_extract(self.config.metadata_url)
        print(f'Using metadata file: {metadata_file}')

        # Download data
        # NOTE: The extracted path contains a subfolder, data_dir. This directory holds
        # a large number of json files (one json file per patent application).
        download_dir = ""
        if self.config.metadata_url[0:5] == "https":
            download_dir = dl_manager.download_and_extract(self.config.data_url)
        json_dir = os.path.join(download_dir, self.config.data_dir)

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
        if self.config.ipcr_label:

            print(f'Filtering by IPCR label: {self.config.ipcr_label}')
            df = df[df['main_ipcr_label'].str.startswith(self.config.ipcr_label)]
        elif self.config.cpc_label:
            print(f'Filtering by CPC label: {self.config.cpc_label}')
            df = df[df['main_cpc_label'].str.startswith(self.config.cpc_label)]

        # Filter metadata based on arbitrary query string
        if self.config.query_string:
            df = df.query(self.config.query_string)
            
        if self.config.force_extract:
            if self.config.name == 'all':
                if self.config.train_filing_start_date and self.config.val_filing_end_date:
                    if self.config.train_filing_end_date and self.config.val_filing_start_date:
                        training_year_range = set(range(int(self.config.train_filing_start_date[:4]), int(self.config.train_filing_end_date[:4]) + 1))
                        validation_year_range = set(range(int(self.config.val_filing_start_date[:4]), int(self.config.val_filing_end_date[:4]) + 1))
                        full_year_range = training_year_range.union(validation_year_range)
                    else:
                        full_year_range = set(range(int(self.config.train_filing_start_date[:4]), int(self.config.val_filing_end_date[:4]) + 1))
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

        if self.config.name == 'all':
            final_df = pd.DataFrame()
            for file in os.listdir(json_dir):
                year = file.split(".")
                # if len(year) == 1 and year[0] in ['2010', '2012','2013','2014', '2015', '2016','2017']:
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
        if self.config.uniform_split:

            # Assumes that training_start_data < val_end_date
            if self.config.train_filing_start_date:
                df = df[df['filing_date'] >= self.config.train_filing_start_date]
            if self.config.val_filing_end_date:
                df = df[df['filing_date'] <= self.config.val_filing_end_date]
            df = df.sample(frac=1.0, random_state=RANDOM_STATE)
            num_train_samples = int(len(df) * 0.85)
            train_df = df.iloc[0:num_train_samples]
            val_df = df.iloc[num_train_samples:-1]

        else:

            # Check
            if not (self.config.train_filing_start_date and self.config.train_filing_end_date and
                    self.config.val_filing_start_date and self.config.train_filing_end_date):
                raise ValueError("Please either use uniform_split or specify your exact \
                    training and validation split dates.")

            # Does not assume that training_start_data < val_end_date
            print(f'Filtering train dataset by filing start date: {self.config.train_filing_start_date}')
            print(f'Filtering train dataset by filing end date: {self.config.train_filing_end_date}')
            print(f'Filtering val dataset by filing start date: {self.config.val_filing_start_date}')
            print(f'Filtering val dataset by filing end date: {self.config.val_filing_end_date}')
            train_df = df[
                (df['filing_date'] >= self.config.train_filing_start_date) & 
                (df['filing_date'] < self.config.train_filing_end_date)
            ]
            val_df = df[
                (df['filing_date'] >= self.config.val_filing_start_date) & 
                (df['filing_date'] < self.config.val_filing_end_date)
            ]

        # TODO: We can probably make this step faster
        if self.config.val_set_balancer:
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

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs=dict(  # these kwargs are passed to _generate_examples
                    df=train_df,
                    json_dir=json_dir,
                    split='train',
                ),
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs=dict(
                    df=val_df,
                    json_dir=json_dir,
                    split='val',
                ),
            ),
        ]

    def _generate_examples(self, df, json_dir, split):
        """ Yields examples by loading JSON files containing patent applications. """

        # NOTE: df.itertuples() is way faster than df.iterrows()
        for id_, x in enumerate(df.itertuples()):

            # JSON files are named by application number (unique)
            application_year = str(x.filing_date.year)
            application_number = x.application_number
            filepath = os.path.join(json_dir, application_year, application_number + '.json')
            try:
                with open(filepath, 'r') as f:
                    patent = json.load(f)
            except Exception as e:
                print('------------')
                print(f'ERROR WITH {filepath}\n')
                print(repr(e))
                print()
                yield id_, {k: "error" for k in _FEATURES}

            # Most up-to-date-decision in meta dataframe
            decision = x.decision
            yield id_, {
                "patent_number": application_number,
                "decision": patent["decision"], # decision,
                "title": patent["title"],
                "abstract": patent["abstract"],
                "claims": patent["claims"],
                "description": patent["full_description"],
                "background": patent["background"],
                "summary": patent["summary"],
                "cpc_label": patent["main_cpc_label"],
                'filing_date': patent['filing_date'],
                'patent_issue_date': patent['patent_issue_date'],
                'date_published': patent['date_published'],
                'examiner_id': patent['examiner_id'],
                "ipc_label": patent["main_ipcr_label"],
            }
