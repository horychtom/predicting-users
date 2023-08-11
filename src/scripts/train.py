from config import WANDB_API_KEY
from src.train.trainer import Trainer
from src.model.model import HFmodel
from src.data.dataset import DataSet
import wandb

training_args = {'lr': 4e-5, 'epochs': 3,
                 'warmup': 10, 'batch_size': 2, 'eval_steps': 20, 'logging_steps': 10}

wandb.login(key=WANDB_API_KEY)
wandb.init(project="framework-test", name="test1")

model = HFmodel(checkpoint='distilbert-base-multilingual-cased')
ds = DataSet('./datasets/binarized.csv', model.tokenizer,
             {'seo_title': 'text', 'total_pageviews': 'labels'})
trainer = Trainer(training_args=training_args, dataset=ds, model=model)
model = trainer.train()
wandb.finish()
