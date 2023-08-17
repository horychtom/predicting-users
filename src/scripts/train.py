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
    num_train_epochs=5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    do_eval=True,
    evaluation_strategy='steps',
    logging_steps=20,
    eval_steps=50,
    disable_tqdm=False,
    weight_decay=0.1,
    warmup_steps=10,
    learning_rate=4e-5,
    run_name='something')


# regression on titles
training_args.run_name = 'all_all_regression'
model = HFmodel(checkpoint='xlm-roberta-base',num_classes=1)
ds = DataSet('datasets/all_all.csv', model.tokenizer,
             {'seo_title': 'text', 'y': 'labels'}, use_dataloaders=False)
trainer = TrainerWrapper(training_args=training_args, dataset=ds, model=model,
                         project_name="testing_data_modelling", run_name=training_args.run_name)
trainer.train()
wandb.finish()

training_args.run_name = 'all_all_regression_on_merged'
model = HFmodel(checkpoint='xlm-roberta-base',num_classes=1)
ds = DataSet('datasets/all_all.csv', model.tokenizer,
             {'merged': 'text', 'total_pageviews': 'labels'}, use_dataloaders=False)
trainer = TrainerWrapper(training_args=training_args, dataset=ds, model=model,
                         project_name="testing_data_modelling", run_name=training_args.run_name)
trainer.train()
wandb.finish()

training_args.run_name = 'all_all_clf'
model = HFmodel(checkpoint='xlm-roberta-base',num_classes=3)
ds = DataSet('datasets/all_all.csv', model.tokenizer,
             {'seo_title': 'text', 'y_disc': 'labels'}, use_dataloaders=False)
trainer = TrainerWrapper(training_args=training_args, dataset=ds, model=model,
                         project_name="testing_data_modelling", run_name=training_args.run_name)
trainer.train()
wandb.finish()

training_args.run_name = 'all_all_clf_merged'
model = HFmodel(checkpoint='xlm-roberta-base',num_classes=3)
ds = DataSet('datasets/all_all.csv', model.tokenizer,
             {'merged': 'text', 'y_disc': 'labels'}, use_dataloaders=False)
trainer = TrainerWrapper(training_args=training_args, dataset=ds, model=model,
                         project_name="testing_data_modelling", run_name=training_args.run_name)
trainer.train()
wandb.finish()