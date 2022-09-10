from PIL import Image

import torch
import torchvision
from torch import nn

import pytorch_lightning as pl
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics import Accuracy, F1Score, ConfusionMatrix
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


class AgePredictResnet(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.model = torchvision.models.resnet101(pretrained=True)
        self.model.fc = nn.Linear(2048, hparams.layer1)
        self.age_linear1 = nn.Linear(hparams.layer1, hparams.layer2)
        self.age_linear2 = nn.Linear(hparams.layer2, hparams.layer3)
        self.age_out = nn.Linear(hparams.layer3, hparams.age_num_targets)
        self.gender_linear1 = nn.Linear(hparams.layer1, hparams.layer2)
        self.gender_linear2 = nn.Linear(hparams.layer2, hparams.layer3)
        self.gender_out = nn.Linear(hparams.layer3, hparams.gender_num)
        self.race_linear1 = nn.Linear(hparams.layer1, hparams.layer2)
        self.race_linear2 = nn.Linear(hparams.layer2, hparams.layer3)
        self.race_out = nn.Linear(hparams.layer3, hparams.race_num)
        if hparams.activation == 'ReLU':
            self.activation = nn.ReLU()
        elif hparams.activation == 'tanh':
            self.activation = nn.Tanh()
        self.dropout = nn.Dropout(hparams.dropout_val)

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
    def __init__(self, config):
        super().__init__()
        # self.save_hyperparameters()
        # self.save_hyperparameters(ignore=[age_weights, race_weights, sex_weights])
        self.default_hparams = config
        age_weights = torch.tensor([0.7704, 1.5936, 0.3426, 0.6154, 1.2710, 1.2029, 2.2643, 3.8366, 4.7582], dtype=torch.float32)
        sex_weights = torch.tensor([0.9581, 1.0458], dtype=torch.float32)
        race_weights = torch.tensor([0.4716, 1.0567, 1.3464, 1.1974, 2.8132], dtype=torch.float32)
        self.model = AgePredictResnet(self.default_hparams)
        self.age_criterion = nn.CrossEntropyLoss(weight=age_weights)
        self.race_criterion = nn.CrossEntropyLoss(weight=race_weights)
        self.sex_criterion = nn.CrossEntropyLoss(weight=sex_weights)
        self.acc = Accuracy()
        self.age_f1 = F1Score(self.default_hparams.age_num_targets, average='macro', mdmc_average='global')
        self.sex_f1 = F1Score(self.default_hparams.gender_num, average='macro', mdmc_average='global')
        self.race_f1 = F1Score(self.default_hparams.race_num, average='macro', mdmc_average='global')
        self.age_confusion = ConfusionMatrix(self.default_hparams.age_num_targets)
        self.sex_confusion = ConfusionMatrix(self.default_hparams.gender_num)
        self.race_confusion = ConfusionMatrix(self.default_hparams.race_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image, transforms):
        image = transforms(image)
        image = torch.unsqueeze(image, 0)
        self.model.eval()
        with torch.inference_mode():
            logits = self.model(image)
        age_prob = self.softmax(logits[0])
        sex_prob = self.softmax(logits[1])
        race_prob = self.softmax(logits[2])
        top2_age = torch.topk(age_prob, 2, dim=1)
        sex = torch.argmax(sex_prob, dim=1)
        top2_race = torch.topk(race_prob, 2, dim=1)
        return (list(top2_age.values.numpy().reshape(-1)), list(top2_age.indices.numpy().reshape(-1))), (sex.item(), sex_prob[0][sex.item()].item()), \
               (list(top2_race.values.numpy().reshape(-1)), list(top2_race.indices.numpy().reshape(-1)))

    def training_step(self, input_batch, batch_idx):
        image_tensors = input_batch[0]
        age_targets = input_batch[1]
        sex_targets = input_batch[2]
        race_targets = input_batch[3]

        logits = self.model(image_tensors)
        age_loss = self.age_criterion(logits[0], age_targets)
        sex_loss = self.sex_criterion(logits[1], sex_targets)
        race_loss = self.race_criterion(logits[2], race_targets)

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

        age_loss = self.age_criterion(logits[0], age_targets)
        sex_loss = self.sex_criterion(logits[1], sex_targets)
        race_loss = self.race_criterion(logits[2], race_targets)

        total_loss = age_loss + sex_loss + race_loss

        age_predict = torch.argmax(logits[0], dim=1)
        sex_predict = torch.argmax(logits[1], dim=1)
        race_predict = torch.argmax(logits[2], dim=1)
        age_acc = self.acc(age_predict, age_targets)
        sex_acc = self.acc(sex_predict, sex_targets)
        race_acc = self.acc(race_predict, race_targets)

        self.log("val-total-loss", total_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val-age-acc', age_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val-sex-acc', sex_acc, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val-race-acc', race_acc, on_step=False, on_epoch=True, prog_bar=False)

        val_dict ={
            'age_predict': age_predict,
            'age_targets': age_targets,
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

        total_f1_score = (age_f1_score + sex_f1_score + race_f1_score)/3

        self.log('age-f1score', age_f1_score)
        self.log('sex-f1score', sex_f1_score)
        self.log('race-f1score', race_f1_score)
        self.log('total-f1score', total_f1_score)

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
        if self.default_hparams.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.default_hparams.lr)
        elif self.default_hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.default_hparams.lr)
        return optimizer
        # scheduler = {
        #             'scheduler': OneCycleLR(optimizer, max_lr=7e-5, steps_per_epoch=304, pct_start=0.15, epochs=30,
        #                        anneal_strategy='cos', div_factor=100),
        #             'interval': 'step'
        #         }
        # return [optimizer], [scheduler]


if __name__ == '__main__':
    trained_model_path = './lightning_logs/resnet101_weighted_no_scheduling/17daip11/checkpoints/epoch=11-total-f1score=0.748.ckpt'
    age_weight = torch.tensor([0.7704, 1.5936, 0.3426, 0.6154, 1.2710, 1.2029, 2.2643, 3.8366, 4.7582], dtype=torch.float32)
    sex_weight = torch.tensor([0.9581, 1.0458], dtype=torch.float32)
    race_weight = torch.tensor([0.4716, 1.0567, 1.3464, 1.1974, 2.8132], dtype=torch.float32)
    model = AgePrediction.load_from_checkpoint(trained_model_path, age_weights=age_weight, sex_weights=sex_weight, race_weights=race_weight)
    sample_image = Image.open('./data/wild_images/part1/50_0_0_20170103183532811.jpg')
    transforms = Compose([Resize((256, 256)), ToTensor(),
                          Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    predictions = model(sample_image, transforms)
    print(predictions)
