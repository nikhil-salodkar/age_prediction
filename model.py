from PIL import Image

import torch
import torchvision
from torch import nn

import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


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
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image, transforms):
        image = transforms(image)
        image = torch.unsqueeze(image, 0)
        print('the image dimensions are :', image.shape)
        with torch.no_grad():
            logits = self.model(image)
        print("the logits are:", logits)
        pred_probabilities = self.softmax(logits)
        print("The probabilities : ", pred_probabilities)
        prediction = torch.argmax(pred_probabilities, dim=1)
        return prediction

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

    def load_previous_trained_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))


if __name__ == '__main__':
    trained_model_path = './lightning_logs/resnet101_64_low_lr_batch_normalized_updated_augmented_adamw/4fg9axib' \
                         '/checkpoints/epoch=22-val-acc=0.828.ckpt'
    model = AgePrediction.load_from_checkpoint(trained_model_path)
    sample_image = Image.open('data/val/1ci33.jpg')
    transforms = Compose([Resize((256, 256)), ToTensor(),
                          Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # model.load_previous_trained_model(trained_model_path)

    single_prediction = model(sample_image, transforms)
    print(single_prediction)
