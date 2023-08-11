import warnings
import logging
import torch
import wandb


from datasets import load_metric
from transformers import DataCollatorWithPadding
import transformers

import numpy as np

from tqdm.auto import tqdm


transformers.logging.set_verbosity(transformers.logging.ERROR)

logging.disable(logging.ERROR)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

BATCH_SIZE = 64


class Trainer():

    def __init__(self, training_args, dataset, model):
        self.lr = training_args['lr']
        self.epochs = training_args['epochs']
        self.warmup = training_args['warmup']
        self.batch_size = training_args['batch_size']
        self.eval_steps = training_args['eval_steps']
        self.logging_steps = training_args['logging_steps']

        self.tokenizer = model.tokenizer
        self.device = model.device
        self.model = model.model

        self.dataset = dataset
        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer)

    def train(self):
        dl = self.dataset.train
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        loss = torch.nn.CrossEntropyLoss()
        loss_sum = 0

        self.model.train()

        for epoch in tqdm(range(self.epoch)):
            for i, batch in tqdm(enumerate(dl)):
                optimizer.zero_grad()
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss
                loss_sum += loss.item()
                loss.backward()
                optimizer.step()

                if (i+1) % self.eval_steps == 0:
                    f1, dev_loss = self.eval(split='eval')
                    wandb.log({'dev_loss': dev_loss, 'dev_f1': f1})
                    self.model.train()

                if (i+1) % self.logging_steps == 0:
                    wandb.log({'train_loss': loss_sum/self.logging_steps})
                    loss_sum = 0

        return self.model

    def eval(self, split: 'test'):
        metric1 = load_metric("f1")

        dl = self.dataset.test if split == 'test' else self.dataset.dev

        loss_sum = 0
        self.model.eval()
        for batch in tqdm(dl):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            logits = outputs.logits
            loss_sum += outputs.loss.item()
            predictions = torch.argmax(logits, dim=-1)
            metric1.add_batch(predictions=predictions,
                              references=batch["labels"])

        return metric1.compute(average='macro')['f1'], loss_sum/(self.batch_size*len(dl))
