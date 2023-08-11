import pandas as pd
from datasets import Dataset
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader


class DataSet():

    def __init__(self, path_to_csv, tokenizer, var_to_text_mapping):
        self.df = pd.read_csv(path_to_csv)
        self.df = self.df.rename(columns=var_to_text_mapping)
        self.tokenizer = tokenizer
        self.dc = DataCollatorWithPadding(tokenizer=self.tokenizer,padding=True)
        self.tok = self.tokenize_data()
        self.train, self.dev, self.test = self.get_dataloaders()

    def tokenize_data(self):
        return pd.DataFrame(dict(self.tokenizer(list(self.df['text']), truncation=True, padding=True, max_length=128)))

    def get_dataloaders(self):
        dataset = self.tok
        dataset['labels'] = self.df['labels']
        dataset = dataset.sample(frac=1)

        train = Dataset.from_dict(dataset[:int(len(dataset)*0.7)])
        dev = Dataset.from_dict(dataset[int(len(dataset)*0.7):int(len(dataset)*0.85)])
        test = Dataset.from_dict(dataset[int(len(dataset)*0.85):])

        train_dl = DataLoader(train, batch_size=32, collate_fn=self.dc)
        dev_dl = DataLoader(dev, batch_size=32, collate_fn=self.dc)
        test_dl = DataLoader(test, batch_size=32, collate_fn=self.dc)

        return train_dl, dev_dl, test_dl
