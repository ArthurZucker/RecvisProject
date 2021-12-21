import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import VOCSegmentation


class VOCSegmentationDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = self.config.batch_size

    # When doing distributed training, Datamodules have two optional arguments for
    # granular control over download/prepare/splitting data:

    # OPTIONAL, called only on 1 GPU/machine
    def prepare_data(self):
        VOCSegmentation(self.config.asset_path, train=True, download=True)
        VOCSegmentation(self.config.asset_path, train=False, download=True)

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage=None):
        # transforms
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        # split dataset
        if stage in (None, "fit"):
            init_dataset = VOCSegmentation(
                self.config.asset_path, train=True, transform=transform
            )
            # Split between train and valid set (80/20)
            lengths = [int(len(init_dataset)*self.config.split_val, int(len(init_dataset)*(1-self.config.split_val)))]            
            self.voc_train, self.voc_val = random_split(init_dataset, lengths)
        if stage == "test":
            self.voc_test = VOCSegmentation(
                self.config.asset_path, train=False, transform=transform
            )
        if stage == "predict":
            # during prediction, the logginf is disabled
            self.voc_predict = VOCSegmentation(
                self.config.asset_path, train=False, transform=transform
            )

    # return the dataloader for each split
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
