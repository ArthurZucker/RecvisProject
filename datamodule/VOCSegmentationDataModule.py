import os

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import VOCSegmentation


class VOCSegmentationDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = self.config.batch_size
        self.root = os.path.join(self.config.asset_path,"VOC")
        self.transform = transforms.Compose(
            [
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.target_transform = transforms.Compose(
            [
                transforms.Resize((256,256)),
                transforms.ToTensor(),
            ]
        )
    # When doing distributed training, Datamodules have two optional arguments for
    # granular control over download/prepare/splitting data:

    # OPTIONAL, called only on 1 GPU/machine
    def prepare_data(self):
        
        VOCSegmentation(root = self.root, image_set='trainval', download=False)
        VOCSegmentation(root = self.root, image_set='val', download=False)

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage=None):
        # transforms
        # split dataset
        if stage in (None, "fit"):
            init_dataset = VOCSegmentation(
                self.root, image_set='trainval', transform=self.transform, target_transform=self.target_transform
            )
            # Split between train and valid set (80/20)
            val_length =int(len(init_dataset)*self.config.split_val)
            lengths = [val_length, len(init_dataset)-val_length]            
            self.voc_train, self.voc_val = random_split(init_dataset, lengths)
        if stage == "test":
            self.voc_test = VOCSegmentation(
                self.root, image_set='val', transform=self.transform, target_transform=self.target_transform
            )
        if stage == "predict":
            # during prediction, the logginf is disabled
            self.voc_predict = VOCSegmentation(
                self.root, image_set='val', transform=self.transform, target_transform=self.target_transform
            )
    # TODO num_workers
    # return the dataloader for each split
    # TODO overwrite get item to cast input masks to torch.LongTensor!!!!!
    
    def train_dataloader(self):
        voc_train = DataLoader(self.voc_train, batch_size=self.batch_size)
        return voc_train

    def val_dataloader(self):
        voc_val = DataLoader(self.voc_val, batch_size=self.batch_size)
        return voc_val

    def test_dataloader(self):
        voc_test = DataLoader(self.voc_test, batch_size=self.batch_size)
        return voc_test

    def predict_dataloader(self):
        voc_predict = DataLoader(self.voc_predict, batch_size=self.batch_size)
        return voc_predict
