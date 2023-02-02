import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl

import wandb
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from dataset import AgePredictionData
from model import AgePrediction


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
    return age_class_weights, sex_class_weights, race_class_weights


def train_model(train_module, data_module):
    """This function can be called when training is not done through hyperparameter sweeps"""
    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{total-f1score:.3f}', save_top_k=2, monitor='total-f1score'
                                          , mode='max')
    early_stopping = EarlyStopping(monitor="total-f1score", patience=5, verbose=False, mode="max")
    # lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(accelerator='gpu', fast_dev_run=False, max_epochs=30,
                         callbacks=[checkpoint_callback, early_stopping], precision=16)
    trainer.fit(train_module, data_module)


def sweep_train():
    wandb_logger = WandbLogger(project="UTK_Age_Prediction", save_dir='./lightning_logs')
    print("the config being used is :", wandb.config)

    pl.seed_everything(7)
    full_df = get_data(wandb.config.path)
    age_weights, sex_weights, race_weights = compute_class_weights(full_df)
    wandb.config['age_weights'] = age_weights
    wandb.config['sex_weights'] = sex_weights
    wandb.config['race_weights'] = race_weights

    train_module = AgePrediction(**wandb.config)
    data_module = AgePredictionData(full_df, **wandb.config)
    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{total-f1score:.3f}', save_top_k=2, monitor='total-f1score'
                                          , mode='max')
    early_stopping = EarlyStopping(monitor="total-f1score", patience=5, verbose=False, mode="max")
    # lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(accelerator='gpu', fast_dev_run=False, max_epochs=wandb.config.epochs,
                         callbacks=[checkpoint_callback, early_stopping], logger=wandb_logger, precision=16)
    trainer.fit(train_module, data_module)


if __name__ == '__main__':
    sweep_train()
