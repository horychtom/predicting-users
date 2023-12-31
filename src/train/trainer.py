import warnings
import logging
import torch
import wandb
from config import WANDB_API_KEY, RANDOM_SEED
from src.myutils import set_random_seed


from torch.utils.data import DataLoader
from datasets import load_metric
from transformers import DataCollatorWithPadding, get_scheduler, Trainer
import transformers

import numpy as np
from sklearn.metrics import f1_score, mean_squared_error, r2_score

from tqdm.auto import tqdm

transformers.logging.set_verbosity(transformers.logging.ERROR)
logging.disable(logging.ERROR)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class CustomTrainer():

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
        lr_scheduler = get_scheduler(name="linear",
                                     optimizer=optimizer,
                                     num_warmup_steps=0,
                                     num_training_steps=self.epochs*len(dl))
        loss = torch.nn.CrossEntropyLoss()
        loss_sum = 0

        self.model.train()

        for epoch in tqdm(range(self.epochs)):
            for i, batch in tqdm(enumerate(dl)):
                optimizer.zero_grad()
                batch = {k: v.to(self.device) for k, v in batch.items()}

                outputs = self.model(**batch)
                loss = outputs.loss
                loss_sum += loss.item()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

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


class TrainerWrapper():

    def __init__(self, training_args, dataset, model, project_name, run_name):
        wandb.login(key=WANDB_API_KEY)
        wandb.init(project=project_name, name=run_name)
        self.training_args = training_args
        self.tokenizer = model.tokenizer
        self.device = model.device
        self.num_classes = model.num_classes
        self.model = model.model
        self.dataset = dataset
        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer)
        self.trainer = Trainer(self.model,
                               training_args,
                               train_dataset=self.dataset.train,
                               eval_dataset=self.dataset.dev,
                               data_collator=self.data_collator,
                               tokenizer=self.tokenizer,
                               compute_metrics=self.compute_metrics)

    def train(self):
        self.model.train()
        self.trainer.train()
        wandb.log({"training": 0})

    def compute_metrics(self, eval_preds):
        predictions = eval_preds.predictions
        targets = eval_preds.label_ids

        if self.num_classes == 1:  # regression
            rmse = mean_squared_error(targets, predictions, squared=False)
            r2 = r2_score(targets, predictions)
            return {"rmse": rmse, "r2": r2}

        predictions = predictions.argmax(axis=1)
        return {'f1': f1_score(targets, predictions, average='macro')}

    def compute_metrics_deprecated(self, dl):
        metric1 = load_metric("f1")

        loss_sum = 0
        self.model.eval()
        for batch in dl:
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            logits = outputs.logits
            loss_sum += outputs.loss.item()
            predictions = torch.argmax(logits, dim=-1)
            metric1.add_batch(predictions=predictions,
                              references=batch["labels"])

        self.model.train()

        return {'f1': metric1.compute(average='macro')['f1']}

    def eval_test(self):
        dl = DataLoader(
            self.dataset.test, batch_size=self.training_args.per_device_eval_batch_size, collate_fn=self.data_collator)
        wandb.log({'test_f1': self.compute_metrics_deprecated(dl=dl)})
        wandb.finish()
