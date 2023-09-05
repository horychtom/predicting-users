
from src.storage import storage_client

import pandas as pd
import transformers

from transformers import DataCollatorWithPadding, RobertaConfig, RobertaTokenizer, AutoTokenizer
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from nltk import sent_tokenize
import nltk
nltk.download('punkt')
tqdm.pandas()


# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("MiriUll/distilbert-german-text-complexity")
model = AutoModelForSequenceClassification.from_pretrained("MiriUll/distilbert-german-text-complexity")

device = torch.device("cuda")
model.to(device)
model.eval()

class ModelInference:

    def __init__(self):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.collator=DataCollatorWithPadding(tokenizer=self.tokenizer)
        # self.mapping = {0: "easy_language",1: "plain_language",2: "everyday_language",3: "special_language"}

    def tokenize(self,x):
        tok = self.tokenizer(x, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        tok["input_ids"] = tok["input_ids"].squeeze()
        tok["attention_mask"] = tok["attention_mask"].squeeze()
        return tok

    def classify_sentence(self,inputs):
        dl = DataLoader(inputs, batch_size=128, collate_fn=self.collator)
        outputs = []
        for batch in dl:
            with torch.no_grad():
                batch.to(self.device)
                model_output = self.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                # outputs.extend(F.softmax(model_output.logits,dim=1)[:,1].tolist())
                return model_output.logits[0].item()
                # outputs.extend(F.softmax(model_output.logits,dim=1).argmax(dim=1).tolist())

        return outputs

    def classify_body(self,body,id):
        annotations = self.classify_sentence([self.tokenize(body)])
        return {'id':id,'text':body,'complexity':annotations}



storage_client.download_from_gcs_to_local_directory_or_file('.', 'datasets/datasets_final.csv')
df = pd.read_csv('datasets/datasets_final.csv')

mi = ModelInference()

rowlist=[]
from tqdm import tqdm
for idx,row in tqdm(df.iterrows()):
    rowlist.append(mi.classify_body(row.seo_title,row.id))
imd = pd.DataFrame(rowlist)
imd.to_csv('datasets/complexity2.csv')

storage_client.upload_local_directory_or_file_to_gcs('datasets/complexity2.csv', 'datasets/complexity2.csv')
