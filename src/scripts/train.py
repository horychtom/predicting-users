from config import WANDB_API_KEY
from src.train.trainer import TrainerWrapper
from src.model.model import HFmodel
from src.data.dataset import DataSet
import wandb

from transformers import TrainingArguments


training_args = TrainingArguments(
    report_to='wandb',
    output_dir='./',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    do_eval=True,
    evaluation_strategy='steps',
    logging_steps=2,
    eval_steps=2,
    disable_tqdm=False,
    weight_decay=0.1,
    warmup_steps=10,
    learning_rate=4e-5,
    run_name='novy_run')

wandb.login(key=WANDB_API_KEY)
wandb.init(project="frameworkie", name="test1")

model = HFmodel(checkpoint='xlm-roberta-base')
ds = DataSet('./datasets/babe.csv', model.tokenizer,
             {'text': 'text', 'label': 'labels'}, use_dataloaders=False)
trainer = TrainerWrapper(training_args=training_args, dataset=ds, model=model)
trainer.train()
wandb.finish()
