"""Data loader for the bird dataset provided by the TA and modified to fit my repository's architecture

"""
import torchvision
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
from torchvision import datasets
from torchvision.transforms import transforms
from utils.transforms import SemanticSegmentation
import torch
import numpy as np

from sklearn.model_selection import KFold


class BirdsDataloader():
    """
    Creates a dataloader for train and val splits
    """

    def __init__(self, args):
        self.config = args
        self.transform = {"train": transforms.Compose([


            # transforms.Resize((520, 520)),

            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.uint8),
            T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
            transforms.ConvertImageDtype(torch.float),
            # T.RandomPerspective(distortion_scale=0.6, p=1.0),

            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),

            # transforms.RandomErasing(inplace=True),


            # SemanticSegmentation(),
            transforms.Resize((384, 384)),
        ]),
            "val": transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ]
        )
        }

        self.train_dataset = datasets.ImageFolder(
            self.config.image_dir + '/train_images', transform=self.transform["train"])
        self.valid_dataset = datasets.ImageFolder(
            self.config.image_dir + '/val_images', transform=self.transform["val"])
        
        self.init_data_loaders()
        #  self.visualize_data()

    def init_data_loaders(self):
        """Initialize data_loaders using the defined datasets
        """
        if self.config.weighted_sampler:
            class_sample_count = np.array([len(np.where(self.train_dataset.targets==t)[0]) for t in np.unique(self.train_dataset.targets)])
            weight = (1 / torch.Tensor(class_sample_count))
            samples_weight = np.array([weight[t] for t in self.train_dataset.targets])
            sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, num_workers=self.config.num_workers,sampler=sampler)
        
        else:
            self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers,drop_last=True)
        self.valid_loader = DataLoader(
            self.valid_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers,drop_last=True)

        self.train_iterations = len(self.train_loader)
        self.valid_iterations = len(self.valid_loader)

    def visualize_data(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import wandb
        idx2class = {v: k for k, v in self.train_dataset.class_to_idx.items()}
        def get_class_distribution(dataset_obj):
            count_dict = {k:0 for k,v in dataset_obj.class_to_idx.items()}
            
            for element in dataset_obj:
                y_lbl = element[1]
                y_lbl = idx2class[y_lbl]
                count_dict[y_lbl] += 1
                    
            return count_dict
        
        plt.figure(figsize=(15,8))
        plt.ioff()
        sns.barplot(data = pd.DataFrame.from_dict([get_class_distribution(self.train_dataset)]).melt(), x = "variable", y="value", hue="variable").set_title('Train Class Distribution')
        plt1 = wandb.Image(plt)
        plt.figure(figsize=(15,8))
        sns.barplot(data = pd.DataFrame.from_dict([get_class_distribution(self.valid_dataset)]).melt(), x = "variable", y="value", hue="variable").set_title('Val Images Class Distribution')
        plt2 = wandb.Image(plt)
        wandb.log({"Distributions : ": [plt1,plt2]})
        plt.close()