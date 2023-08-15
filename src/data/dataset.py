import pandas as pd
from datasets import Dataset
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
import os
from src.storage import storage_client
from config import RANDOM_SEED


class DataSet():

    def __init__(self, path_to_csv, tokenizer, var_to_text_mapping, use_dataloaders: bool):
        self.df = self.download_data(path_to_csv)
        self.df = self.df.rename(columns=var_to_text_mapping)
        self.tokenizer = tokenizer
        self.dc = DataCollatorWithPadding(
            tokenizer=self.tokenizer, padding=True)
        self.tok = self.tokenize_data()
        self.train, self.dev, self.test = self.split_dataset(
            use_dataloaders=use_dataloaders)

    def download_data(self, path_to_csv):
        if not os.path.exists(path_to_csv):
            storage_client.download_from_gcs_to_local_directory_or_file(
                local_path="", gcs_path=path_to_csv)
        return pd.read_csv(path_to_csv)

    def tokenize_data(self):
        return pd.DataFrame(dict(self.tokenizer(list(self.df['text']), truncation=True, padding=True, max_length=128)))

    def split_dataset(self, use_dataloaders=False):
        dataset = self.tok
        dataset['labels'] = self.df['labels']
        dataset = dataset.sample(frac=1,random_state=RANDOM_SEED)

        train = Dataset.from_dict(dataset[:int(len(dataset)*0.7)])
        dev = Dataset.from_dict(
            dataset[int(len(dataset)*0.7):int(len(dataset)*0.85)])
        test = Dataset.from_dict(dataset[int(len(dataset)*0.85):])

        if use_dataloaders:
            train = DataLoader(train, batch_size=32, collate_fn=self.dc)
            dev = DataLoader(dev, batch_size=32, collate_fn=self.dc)
            test = DataLoader(test, batch_size=32, collate_fn=self.dc)

        return train, dev, test
