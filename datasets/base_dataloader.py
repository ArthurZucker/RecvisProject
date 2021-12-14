from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from datasets.base_dataset import base_dataset


class base_dataloader():
    """
    Creates a dataloader for train and val splits
    """
    def __init__(self,args):
        self.config = args
        self.transform = None

        if self.config.test_mode:
            self.train_iterations = 10

        print(f"Basic dataloader, data_mode : {args.dataset}, path : {self.config.img_dir}")
        dataset = base_dataset(self.config.img_dir,self.config.annotation_file,self.config.input_dim,self.transform)
            
        train_indices, valid_indices = train_test_split(range(len(dataset)),test_size=self.config.valid_size,train_size=1-self.config.valid_size,
                                                        shuffle=False)

        train_dataset = Subset(dataset, train_indices)
        valid_dataset = Subset(dataset, valid_indices)

        self.len_train_data = len(train_dataset)
        self.len_valid_data = len(valid_dataset)

        self.train_iterations = (self.len_train_data + self.config.batch_size - 1) // self.config.batch_size
        self.valid_iterations = (self.len_valid_data + self.config.batch_size - 1) // self.config.batch_size

        self.train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True,num_workers=self.config.num_workers)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.config.batch_size, shuffle=False,num_workers=self.config.num_workers) 
