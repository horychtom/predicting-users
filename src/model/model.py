from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from torch import nn
from transformers import AutoModel
import torch


class CustomModel(nn.Module):
    """Torch-based module."""

    def __init__(self):
        """Inititialize model and create heads."""
        super().__init__()
        self.language_model = AutoModel.from_pretrained("xlm-roberta-base")
        self.regressor = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.language_model.config.hidden_size, 1))

    def forward(self, input_ids, attention_mask, labels):
        """Pass the data through the model and according head decided from heads dict."""
        # pass through the model
        x_enc = self.language_model(
            input_ids=input_ids, attention_mask=attention_mask).pooler_output
        out = self.regressor(x_enc)
        loss = nn.MSELoss()
        output = loss(out, labels)
        return {'loss': output, 'logits': out}


class HFmodel():

    def __init__(self, checkpoint='xlm-roberta-base') -> None:
        super().__init__()
        self.checkpopint = checkpoint
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)

    def eval(self, eval_=True):
        if eval_:
            self.model.eval()
        else:
            self.model.train()
