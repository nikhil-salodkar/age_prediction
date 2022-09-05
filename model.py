from PIL import Image

import torch
import torchvision
from torch import nn

import pytorch_lightning as pl
from torch.optim.lr_scheduler import OneCycleLR
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
        self.age_f1 = F1Score(num_classes=9, average='macro', mdmc_average='global')
        self.sex_f1 = F1Score(num_classes=2, average='macro', mdmc_average='global')
        self.race_f1 = F1Score(num_classes=5, average='macro', mdmc_average='global')
        self.age_confusion = ConfusionMatrix(num_classes=9)
        self.sex_confusion = ConfusionMatrix(num_classes=2)
        self.race_confusion = ConfusionMatrix(num_classes=5)
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

        self.log("age-loss", age_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("sex-loss", sex_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("race-loss", race_loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("total-loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('age-acc', age_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log('sex-acc', sex_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log('race-acc', race_acc, on_step=False, on_epoch=True, prog_bar=False)
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

        self.log("val-total-loss", total_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val-age-acc', age_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val-sex-acc', sex_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val-race-acc', race_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val-total-acc', total_acc, on_step=False, on_epoch=True, prog_bar=False)

        val_dict =  {
            'age_predict' : age_predict,
            'age_targets' : age_targets,
            'sex_predict': sex_predict,
            'sex_targets': sex_targets,
            'race_predict': race_predict,
            'race_targets': race_targets
        }

        return val_dict

    def validation_epoch_end(self, validation_outputs):
        age_preds, age_targets, sex_preds, sex_targets = [], [], [], []
        race_preds, race_targets = [], []
        for x in validation_outputs:
            age_preds.append(x['age_predict'])
            age_targets.append(x['age_targets'])
            sex_preds.append(x['sex_predict'])
            sex_targets.append(x['sex_targets'])
            race_preds.append(x['race_predict'])
            race_targets.append(x['race_targets'])
        all_age_preds = torch.stack(age_preds[0:-1]).view(-1)
        all_age_targets = torch.stack(age_targets[0:-1]).view(-1)

        all_sex_preds = torch.stack(sex_preds[0:-1]).view(-1)
        all_sex_targets = torch.stack(sex_targets[0:-1]).view(-1)

        all_race_preds = torch.stack(race_preds[0:-1]).view(-1)
        all_race_targets = torch.stack(race_targets[0:-1]).view(-1)

        all_age_preds = torch.cat((all_age_preds, age_preds[-1:][0]))
        all_age_targets = torch.cat((all_age_targets, age_targets[-1:][0]))

        all_sex_preds = torch.cat((all_sex_preds, sex_preds[-1:][0]))
        all_sex_targets = torch.cat((all_sex_targets, sex_targets[-1:][0]))

        all_race_preds = torch.cat((all_race_preds, race_preds[-1:][0]))
        all_race_targets = torch.cat((all_race_targets, race_targets[-1:][0]))

        age_confusion_metric = self.age_confusion(all_age_preds, all_age_targets)
        sex_confusion_metric = self.sex_confusion(all_sex_preds, all_sex_targets)
        race_confusion_metric = self.race_confusion(all_race_preds, all_race_targets)

        age_f1_score = self.age_f1(all_age_preds, all_age_targets)
        sex_f1_score = self.sex_f1(all_sex_preds, all_sex_targets)
        race_f1_score = self.race_f1(all_race_preds, all_race_targets)

        self.log('age-f1score', age_f1_score)
        self.log('sex-f1score', sex_f1_score)
        self.log('race-f1score', race_f1_score)

        print("age_confusion_metric: ", age_confusion_metric)
        print("sex_confusion_metric: ", sex_confusion_metric)
        print("race_confusion_metric: ", race_confusion_metric)

    def predict_step(self, input_batch, batch_idx):
        print("inside predict_step")
        image_tensors = input_batch[0]
        targets = input_batch[1]

        logits = self.model(image_tensors)

        predictions = torch.argmax(logits, dim=1)
        return predictions.cpu().detach().numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        scheduler = {
                    'scheduler': OneCycleLR(optimizer, max_lr=7e-5, steps_per_epoch=304, pct_start=0.15, epochs=30,
                               anneal_strategy='cos', div_factor=100),
                    'interval': 'step'
                }
        return [optimizer], [scheduler]


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
