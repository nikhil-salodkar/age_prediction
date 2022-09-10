import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from sweep_configuration import sweep_config
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from dataset import AgePredictionData
from model import AgePrediction


default_hparams = {
    'layer1': 512,
    'layer2': 256,
    'layer3': 128,
    'age_num_targets': 9,
    'gender_num': 2,
    'race_num': 5,
    'activation': 'ReLU',
    'batch_size': 64,
    'lr': 2e-5,
    'epochs': 30,
    'dropout_val': 0.4,
    'optimizer': 'adamw',
    'path': './data/wild_images'
}

def get_data(data_path):
    df = pd.read_csv(os.path.join(data_path, 'full_wild_images_new.csv'))
    le = LabelEncoder()
    le.fit(df.age_range)
    print(le.classes_)
    ages = le.transform(df['age_range'])
    df['target_age'] = ages
    return df


def compute_class_weights(df):
    age_class_weights = compute_class_weight('balanced', classes=np.unique(df.target_age), y=df.target_age)
    race_class_weights = compute_class_weight('balanced', classes=np.unique(df.race), y=df.race)
    sex_class_weights = compute_class_weight('balanced', classes=np.unique(df.sex), y=df.sex)
    return torch.tensor(age_class_weights, dtype=torch.float32), torch.tensor(sex_class_weights, dtype=torch.float32), \
           torch.tensor(race_class_weights, dtype=torch.float32)


def train_model(train_module, data_module):
    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{total-f1score:.3f}', save_top_k=2, monitor='total-f1score'
                                          , mode='max')
    early_stopping = EarlyStopping(monitor="total-f1score", patience=5, verbose=False, mode="max")
    # lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(accelerator='gpu', fast_dev_run=False, max_epochs=30,
                         callbacks=[checkpoint_callback, early_stopping], logger=wandb_logger, precision=16)
    trainer.fit(train_module, data_module)


def sweep_train():
    # set up W&B logger
    # wandb.init(name="resnet101_sweep1")
    wandb_logger = WandbLogger(project="UTK_Age_Prediction", save_dir='./lightning_logs')
    print("the config being used is :", wandb.config)

    pl.seed_everything(7)
    full_df = get_data(wandb.config.path)
    # age_weights, sex_weights, race_weights = compute_class_weights(full_df)

    train_module = AgePrediction(wandb.config)
    data_module = AgePredictionData(wandb.config, full_df)
    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{total-f1score:.3f}', save_top_k=2, monitor='total-f1score'
                                          , mode='max')
    early_stopping = EarlyStopping(monitor="total-f1score", patience=5, verbose=False, mode="max")
    # lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(accelerator='gpu', fast_dev_run=False, max_epochs=wandb.config.epochs,
                         callbacks=[checkpoint_callback, early_stopping], logger=wandb_logger, precision=16)
    trainer.fit(train_module, data_module)


if __name__ == '__main__':
    sweep_train()
