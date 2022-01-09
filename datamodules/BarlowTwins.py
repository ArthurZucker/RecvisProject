import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import datasets 

class BarlowTwins(LightningDataModule):
    """Data Module for barlowtwins training

    Args:
        LightningDataModule ([type]): [description]
    """
    def __init__(self, config,dataset_name = "BarlowTwinsDataset"):
        super().__init__()
        self.dataset    = getattr(datasets,dataset_name)
        self.config     = config
        self.batch_size = self.config.batch_size
        self.root = os.path.join(self.config.asset_path, "VOC")

    # def prepare_data(self):
    # #use to download
    #     self.dataset(root = self.root, img_size=self.config.input_size,image_set = "Train")
    #     self.dataset(root = self.root, img_size=self.config.input_size,image_set = "Val")

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage=None):
        # transforms
        # split dataset
        if stage in (None, "fit"):
            self.cifar_train = self.dataset(
                self.root, img_size=self.config.input_size,image_set = "train"
            )
            self.cifar_val = self.dataset(
                self.root, img_size=self.config.input_size,image_set = "val"
            )

    def train_dataloader(self):
        cifar_train = DataLoader(
            self.cifar_train,
            batch_size=self.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
        )
        return cifar_train

    def val_dataloader(self):
        cifar_val = DataLoader(
            self.cifar_val,
            batch_size=self.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
        )
        return cifar_val
