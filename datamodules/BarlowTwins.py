import os

import datasets
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class BarlowTwins(LightningDataModule):
    """Data Module for barlowtwins training

    Args:
        LightningDataModule ([type]): [description]
    """

    def __init__(self, config):
        super().__init__()
        dataset_name = config.hparams.dataset
        self.dataset = getattr(datasets, dataset_name)
        self.config = config.data_param
        self.batch_size = self.config.batch_size
        if self.config.root_dataset is not None:
            self.root = self.config.root_dataset
        else:
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
            self.train = self.dataset(
                self.root, img_size=self.config.input_size, image_set="train"
            )
            self.val = self.dataset(
                self.root, img_size=self.config.input_size, image_set="val"
            )

    def train_dataloader(self):
        train = DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
        )
        return train

    def val_dataloader(self):
        val = DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
        )
        return val
