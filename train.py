import os

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.preprocessing import LabelEncoder

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


def train_model(train_module, data_module):
    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{val-total-acc:.3f}', save_top_k=2, monitor='val-total-acc'
                                          , mode='max')
    early_stopping = EarlyStopping(monitor="val-total-acc", patience=5, verbose=False, mode="max")
    wandb_logger = WandbLogger(project="UTK_Age_Prediction", save_dir='./lightning_logs',
                               name="resnet101_fourth_run")

    trainer = pl.Trainer(accelerator='gpu', fast_dev_run=False, max_epochs=50,
                         callbacks=[checkpoint_callback, early_stopping], logger=wandb_logger, precision=16)
    trainer.fit(train_module, data_module)


if __name__ == '__main__':
    pl.seed_everything(7)
    train_module = AgePrediction()
    path = './data/wild_images'
    full_df = get_data(path)
    data_module = AgePredictionData(full_df, 64, path)
    train_model(train_module, data_module)

