import os
from typing import Optional

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from torchvision.transforms import RandomHorizontalFlip, RandomRotation, ColorJitter, GaussianBlur, RandomGrayscale, \
    Compose, Resize, RandomApply, ToTensor, Normalize


class AgeDataset(Dataset):
    def __init__(self, data, transforms, path='../data/wild_images'):
        self.data = data
        self.transforms = transforms
        self.path = path

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        row = self.data.iloc[index]
        image_id = row['file_path']
        target_age = row['target_age']
        target_sex = row['sex']
        target_race = row['race']
        subfolder = row['subfolder']
        the_image = Image.open(os.path.join(self.path, subfolder, image_id))
        transformed_image = self.transforms(the_image)
        return transformed_image, target_age, target_sex, target_race


class AgePredictionData(pl.LightningDataModule):
    def __init__(self, hparams, full_data):
        super().__init__()
        self.path = hparams.path
        self.full_data = full_data
        self.augmentations = [
            RandomHorizontalFlip(0.5),
            RandomRotation(30),
            ColorJitter(brightness=.5, hue=.3),
            GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            RandomGrayscale()
        ]
        self.train_transforms = Compose([
            Resize((256, 256)),
            RandomApply(self.augmentations, p=0.6),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transforms = Compose([Resize((256, 256)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.train_batch = hparams.batch_size
        self.val_batch = self.train_batch * 4

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            train_data, self.test_data = train_test_split(self.full_data, test_size=0.05,
                                                     stratify=self.full_data['target_age'])
            self.train, self.val = train_test_split(train_data, test_size=0.15, stratify=train_data['target_age'])
            self.train_data = AgeDataset(self.train, self.train_transforms, self.path)
            self.val_data = AgeDataset(self.val, self.transforms, self.path)

        if stage == 'predict':
            self.pred_data = AgeDataset(self.full_data, self.transforms, self.path)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.train_batch, shuffle=True, pin_memory=True,
                          num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.val_batch, shuffle=False, pin_memory=True,
                          num_workers=16)

    def predict_dataloader(self):
        return DataLoader(self.pred_data, batch_size=self.val_batch, shuffle=False, pin_memory=True, num_workers=16)
