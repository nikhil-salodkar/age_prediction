import torch
import torchvision
from torch import nn

import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score


class AgePredictResnet(nn.Module):
    def __init__(self, num_targets=10):
        super().__init__()

        self.model = torchvision.models.resnet101(pretrained=True)
        self.model.fc = nn.Linear(2048, num_targets)

    def forward(self, x):
        out = self.model(x)
        return out


class AgePrediction(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = AgePredictResnet()
        self.criterion = nn.CrossEntropyLoss()
        self.acc = Accuracy()
        self.f1 = F1Score()

    def forward(self, x):
        return self.model(x)

    def training_step(self, input_batch, batch_idx):
        image_tensors = input_batch[0]
        targets = input_batch[1]

        logits = self.model(image_tensors)
        loss = self.criterion(logits, targets)

        predictions = torch.argmax(logits, dim=1)
        the_acc = self.acc(predictions, targets)

        self.log("train-loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train-acc", the_acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, input_batch, batch_idx):
        image_tensors = input_batch[0]
        targets = input_batch[1]

        logits = self.model(image_tensors)

        loss = self.criterion(logits, targets)
        predictions = torch.argmax(logits, dim=1)
        the_acc = self.acc(predictions, targets)

        self.log("val-loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val-acc", the_acc, on_step=False, on_epoch=True, prog_bar=True)

    def predict_step(self, input_batch, batch_idx):
        print("inside predict_step")
        image_tensors = input_batch[0]
        targets = input_batch[1]

        logits = self.model(image_tensors)

        predictions = torch.argmax(logits, dim=1)
        return predictions

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.00002)