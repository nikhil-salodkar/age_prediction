import os

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from dataset import AgePredictionData
from model import AgePrediction

pl.seed_everything(7)


def train_data(the_path):
    train = pd.read_csv(the_path)
    train = train.sample(frac=1).reset_index(drop=True)
    return train


def val_data(the_path):
    val = pd.read_csv(the_path)
    return val


def train_model(train_module, data_module):
    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{val-acc:.3f}', save_top_k=1, monitor='val-acc'
                                          , mode='max')
    early_stopping = EarlyStopping(monitor="val-acc", patience=20, verbose=False, mode="max")
    wandb_logger = WandbLogger(project="Age_Prediction", save_dir='lightning_logs',
                               name="resnet101_64_low_lr_batch_normalized_updated_augmented_adamw")

    trainer = pl.Trainer(accelerator='gpu', fast_dev_run=False, max_epochs=100,
                         callbacks=[checkpoint_callback, early_stopping], logger=wandb_logger, precision=16,
                         log_every_n_steps=25)
    trainer.fit(train_module, data_module)


def only_validate_model(train_module, data_module, model_path):
    wandb_logger = WandbLogger(project="Age_Prediction", save_dir='lightning_logs',
                               name="resnet101_64_testing")
    trainer = pl.Trainer(accelerator='gpu', fast_dev_run=False, precision=16, logger=wandb_logger)
    data_module.setup()
    trainer.validate(train_module, data_module, ckpt_path=model_path)
    # trainer.validate(train_module, data_module)

def get_predictions(train_module, data_module, model_path):
    trainer = pl.Trainer(accelerator='gpu', fast_dev_run=False, precision=16)
    data_module.setup(stage='predict')
    all_preds = trainer.predict(train_module, data_module, ckpt_path=model_path)
    return all_preds


if __name__ == '__main__':
    train_module = AgePrediction()
    path = './data'
    checkpoint_path = './lightning_logs/resnet101_64_low_lr_batch_normalized_updated_augmented_adamw/4fg9axib' \
                      '/checkpoints/epoch=22-val-acc=0.828.ckpt'
    train_images = train_data(os.path.join(path, 'train_new.csv'))
    val_images = val_data(os.path.join(path, 'val_new.csv'))
    data_module = AgePredictionData(train_images, val_images, 64)
    predictions = get_predictions(train_module, data_module)
    # train_model(train_module, data_module)
    # only_validate_model(train_module, data_module, checkpoint_path)
