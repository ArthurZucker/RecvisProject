import os

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from dataset.BarlowTwinsDataset import BarlowTwinsDataset



class BarlowTwinsVOCSegmentationDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = self.config.batch_size
        self.root = os.path.join(self.config.asset_path, "VOC")

    # When doing distributed training, Datamodules have two optional arguments for
    # granular control over download/prepare/splitting data:

    # OPTIONAL, called only on 1 GPU/machine
    # def prepare_data(self):
    # use to download
    # BarlowTwinsDataset(root = self.root, image_set='trainval', download=False)
    # BarlowTwinsDataset(root = self.root, image_set='val', download=False)

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage=None):
        # transforms
        # split dataset
        if stage in (None, "fit"):
            self.voc_train = BarlowTwinsDataset(
                self.root, image_set="trainval"
            )
            self.voc_val = BarlowTwinsDataset(
                self.root,
                image_set="val",
            )

    def train_dataloader(self):
        voc_train = DataLoader(
            self.voc_train,
            batch_size=self.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
        )
        return voc_train

