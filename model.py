from PIL import Image

import torch
import torchvision
from torch import nn

import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, ConfusionMatrix
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


class AgePredictResnet(nn.Module):
    def __init__(self, age_num_targets=9, gender_num=2, race_num=5):
        super().__init__()

        self.model = torchvision.models.resnet101(pretrained=True)
        self.model.fc = nn.Linear(2048, 512)
        self.age_linear1 = nn.Linear(512, 256)
        self.age_linear2 = nn.Linear(256, 128)
        self.age_out = nn.Linear(128, age_num_targets)
        self.gender_linear1 = nn.Linear(512, 256)
        self.gender_linear2 = nn.Linear(256, 128)
        self.gender_out = nn.Linear(128, gender_num)
        self.race_linear1 = nn.Linear(512, 256)
        self.race_linear2 = nn.Linear(256, 128)
        self.race_out = nn.Linear(128, race_num)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        out = self.activation(self.model(x))
        age_out = self.activation(self.dropout((self.age_linear1(out))))
        age_out = self.activation(self.dropout(self.age_linear2(age_out)))
        age_out = self.age_out(age_out)

        gender_out = self.activation(self.dropout((self.gender_linear1(out))))
        gender_out = self.activation(self.dropout(self.gender_linear2(gender_out)))
        gender_out = self.gender_out(gender_out)

        race_out = self.activation(self.dropout((self.race_linear1(out))))
        race_out = self.activation(self.dropout(self.race_linear2(race_out)))
        race_out = self.race_out(race_out)
        return age_out, gender_out, race_out


class AgePrediction(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = AgePredictResnet()
        self.criterion = nn.CrossEntropyLoss()
        self.acc = Accuracy()
        self.f1 = F1Score(num_classes=3, average='macro', mdmc_average='global')
        self.confusion = ConfusionMatrix(num_classes=3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image, transforms):
        print("inside forward")
        image = transforms(image)
        image = torch.unsqueeze(image, 0)
        print('the image dimensions are :', image.shape)
        self.model.eval()
        logits = self.model(image)
        print("the logits are:", logits)
        pred_probabilities = self.softmax(logits)
        print("The probabilities : ", pred_probabilities)
        prediction = torch.argmax(pred_probabilities, dim=1)
        return prediction

    def training_step(self, input_batch, batch_idx):
        image_tensors = input_batch[0]
        age_targets = input_batch[1]
        sex_targets = input_batch[2]
        race_targets = input_batch[3]

        logits = self.model(image_tensors)
        age_loss = self.criterion(logits[0], age_targets)
        sex_loss = self.criterion(logits[1], sex_targets)
        race_loss = self.criterion(logits[2], race_targets)

        total_loss = age_loss + sex_loss + race_loss

        age_predict = torch.argmax(logits[0], dim=1)
        sex_predict = torch.argmax(logits[1], dim=1)
        race_predict = torch.argmax(logits[2], dim=1)
        age_acc = self.acc(age_predict, age_targets)
        sex_acc = self.acc(sex_predict, sex_targets)
        race_acc = self.acc(race_predict, race_targets)

        total_acc = (age_acc + sex_acc + race_acc) / 3

        self.log("age-loss", age_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("sex-loss", sex_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("race-loss", race_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("total-loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('age-acc', age_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('sex-acc', sex_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('race-acc', race_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('total-train-acc', total_acc, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(self, input_batch, batch_idx):
        image_tensors = input_batch[0]
        age_targets = input_batch[1]
        sex_targets = input_batch[2]
        race_targets = input_batch[3]

        logits = self.model(image_tensors)

        age_loss = self.criterion(logits[0], age_targets)
        sex_loss = self.criterion(logits[1], sex_targets)
        race_loss = self.criterion(logits[2], race_targets)

        total_loss = age_loss + sex_loss + race_loss

        age_predict = torch.argmax(logits[0], dim=1)
        sex_predict = torch.argmax(logits[1], dim=1)
        race_predict = torch.argmax(logits[2], dim=1)
        age_acc = self.acc(age_predict, age_targets)
        sex_acc = self.acc(sex_predict, sex_targets)
        race_acc = self.acc(race_predict, race_targets)

        total_acc = (age_acc + sex_acc + race_acc) / 3

        self.log("val-total-loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val-age-acc', age_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val-sex-acc', sex_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val-race-acc', race_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val-total-acc', total_acc, on_step=False, on_epoch=True, prog_bar=True)

    def validation_epoch_end(self, validation_outputs):
        preds, targets = zip(*validation_outputs)
        all_preds = torch.stack(preds[0:-1]).view(-1)
        all_targets = torch.stack(targets[0:-1]).view(-1)

        all_preds = torch.cat((all_preds, preds[-1:][0]))
        all_targets = torch.cat((all_targets, targets[-1:][0]))

        confusion_metric = self.confusion(all_preds, all_targets)
        f1_score = self.f1(all_preds, all_targets)
        self.log('val-f1score', f1_score)

        print("confusion_metric: ", confusion_metric)

    def predict_step(self, input_batch, batch_idx):
        print("inside predict_step")
        image_tensors = input_batch[0]
        targets = input_batch[1]

        logits = self.model(image_tensors)

        predictions = torch.argmax(logits, dim=1)
        return predictions.cpu().detach().numpy()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.00002)


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
