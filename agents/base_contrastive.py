"""
The Base Agent class, where all other agents inherit from, that contains definitions for all the necessary functions
"""
from torchvision import transforms
from utils.misc import print_cuda_statistics
from utils.metrics import AverageMeter, consusion_matrix, compute_metrics, multi_cls_accuracy, multi_cls_roc
from utils.agent_utils import get_loss, get_net, get_optimizer
from utils.feature_visualization import get_representation, plot_contrastive
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
from torch.autograd import Variable
from datasets.BirdsDataloader import BirdsDataloader
from datasets import *
import wandb
import torch.nn as nn
import torch
import shutil
from graphs.losses.Angular import AngularPenaltySMLoss
from os import path

from graphs.losses import SubcenterArcMarginProduct

import numpy as np
np.seterr(divide='ignore', invalid='ignore')
# visualisation tool
# import your classes here


class ContrastiveAgent:
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self, config, run):
        self.config = config
        self.wb_run = run
        self.model = get_net(config)
        print(self.model)
        self.wb_run.watch(self.model)
        self.activation = np.array([])            # run is a wandb instance
        self.feature_hook = self.model.net.fc.register_forward_hook(self.getActivation(f'{self.model.net.fc}'))

        self.data_loader = globals()[self.config.dataloader](self.config)
        self.plot_sample_images()
        # define loss
        self.loss = get_loss(config.loss)
        self.loss = AngularPenaltySMLoss(self.model.num_ftrs,self.config.num_classes)

        # define optimizers for both generator and discriminator
        self.optimizer = get_optimizer(config, self.model)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.8)
        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_valid_acc = 0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            print("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed_all(self.manual_seed)
            torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()
            print("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            print("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)

    def load_checkpoint(self, filename):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        filename = self.config.checkpoint_dir + filename
        try:
            print("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'],strict=False )
            # self.loss.load_state_dict(checkpoint['loss'])

            
            # self.optimizer.load_state_dict(checkpoint['optimizer'])

            print("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                  .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            print("No checkpoint exists from '{}'. Skipping...".format(
                self.config.checkpoint_dir))
            print("**First time to train**")

    def save_checkpoint(self, filename="checkpoint.pth.tar", is_best=0):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'loss': self.loss.state_dict(),
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + filename)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + "/" + filename,
                            self.config.checkpoint_dir + "/" + f'{wandb.run.name}_model_best_{self.best_valid_acc*100:.2f}.pth.tar')

    def run(self):
        """
        The main operator
        :return:
        """
        try:
            if self.config.mode == 'test':
                self.test()
            else:
                self.train()

        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        """
        Main training loop
        :return:
        """

        max_epoch = self.config.max_epoch
        if self.config.test_mode:
            max_epoch = 2

        for epoch in range(self.current_epoch, min(max_epoch, self.config.max_epoch)):
            self.current_epoch = epoch
            self.train_one_epoch()
            self.scheduler.step()
            if epoch % self.config.validate_every == 0:
                valid_acc = self.validate()
                get_representation(self.temb,
                                   self.tpred,
                                   self.vemb,
                                   self.vpred
                                   )
                plot_contrastive(self.temb,
                                   self.tpred,
                                   self.vemb,
                                   self.vpred
                                   )
                self.activation = np.array([])
                is_best = valid_acc > self.best_valid_acc
                if is_best:
                    self.best_valid_acc = valid_acc
                self.save_checkpoint(is_best=is_best)
                self.activation = np.array([])
        self.feature_hook.remove()
        
    def train_one_epoch(self):
        """
        One epoch of training
        uses tqdm to load data in parallel? Nope thats not true
        :return:
        """
        if self.config.test_mode:
            self.data_loader.train_iterations = 10

        tqdm_batch = tqdm(self.data_loader.train_loader, total=self.data_loader.train_iterations,
                          desc="Epoch-{}-".format(self.current_epoch), leave=True)
        # Set the model to be in training mode
        self.model.train()
        epoch_loss = AverageMeter()
        correct = AverageMeter()
        self.tpred = np.array([])
        for current_batch, (x, y) in enumerate(tqdm_batch):
            if self.cuda:
                x, y = x.cuda(non_blocking=self.config.async_loading), y.cuda(
                    non_blocking=self.config.async_loading)
            self.optimizer.zero_grad()
            pred = self.model(x)
            cur_loss = self.loss(pred, y)
            pred = self.loss.pred
            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')
            cur_loss.backward()
            self.optimizer.step()
            epoch_loss.update(cur_loss.item())
            tped = torch.squeeze(torch.argmax(
                pred.detach(), dim=1, keepdim=True))
            correct.update(
                sum(tped == y).cpu() / y.shape[0]
            )

            self.current_iteration += 1
            # logging in wand
            wandb.log({"epoch/loss": epoch_loss.val,
                       "epoch/accuracy": correct.val
                       })

            self.tpred = np.append(self.tpred,tped.cpu().numpy())
            
            if self.config.test_mode and current_batch == 11:
                break
            
            
            
        self.temb = self.activation.copy()
        self.activation = np.array([])
        tqdm_batch.close()
        print("Training at epoch-" + str(self.current_epoch) + " | " + "loss: " + str(
            epoch_loss.val) + "- Top1 Acc: " + str(correct.val))
        
        

    def validate(self):
        """
        One cycle of model validation
        :return:
        """
        if self.config.test_mode:
            self.data_loader.valid_iterations = 5

        tqdm_batch = tqdm(self.data_loader.valid_loader, total=self.data_loader.valid_iterations,
                          desc="Validation at -{}-".format(self.current_epoch))

        # set the model in training mode
        self.model.eval()
        validation_prediction = []
        validation_target = []
        epoch_loss = AverageMeter()
        correct = AverageMeter()
        with torch.no_grad():
            for current_batch, (x, y) in enumerate(tqdm_batch):
                if self.cuda:
                    x, y = x.cuda(non_blocking=self.config.async_loading), y.cuda(
                        non_blocking=self.config.async_loading)
                pred = self.model(x)
                cur_loss = self.loss(pred, y)
                pred = self.loss.pred
                if np.isnan(float(cur_loss.item())):
                    raise ValueError('Loss is nan during validation...')
                epoch_loss.update(cur_loss.item())

                tped = torch.squeeze(torch.argmax(pred, dim=1, keepdim=True))
                correct.update(
                    sum(tped == y).cpu() / y.shape[0]
                )
                dic = {}
                validation_prediction = np.r_[
                    validation_prediction, tped.cpu().numpy()]
                validation_target = np.r_[validation_target, y.cpu().numpy()]

                # dic = compute_metrics(output.cpu(), y.detach().cpu(),self.config.num_classes)
                dic.update({"epoch/validation_loss": epoch_loss.val,
                            "epoch/validation_accuracy": correct.val
                            })
                wandb.log(dic)

                if self.config.test_mode and current_batch == 5:
                    break

            p, r, f, plot = multi_cls_accuracy(
                validation_prediction, validation_target)
            roc_plot, auc = multi_cls_roc(
                validation_prediction, validation_target, self.config.num_classes)
            wandb.log({"RocCurves": [plot, roc_plot], "val/Recall": r,
                       "val/Precision": p, "val/F1": f, "val/mAP": auc})

            wandb.log({"conf_mat": consusion_matrix(
                validation_prediction, validation_target)})

        print("Validation results at epoch-" + str(self.current_epoch)
              + " | " + "loss: " + str(epoch_loss.avg)
              + "\n Top1 Acc \t: " + str(correct.val.item())
              + "\n Precision \t: " + str(p)
              + "\n Recall \t: " + str(r)
              + "\n F1 score \t: " + str(f)
              + "\n mean AP \t: " + str(auc)
              )
        
        tqdm_batch.close()
        self.vemb = self.activation.copy()
        self.vpred = validation_prediction.copy()
        self.activation = np.array([])
        
        return auc #correct.val

    def test(self):
        import PIL.Image as Image

        def pil_loader(path):
            with open(path, 'rb') as f:
                with Image.open(f) as img:
                    return img.convert('RGB')

        # set the model in training mode

        self.model.eval()
        
        output_file = open(self.config.outfile, "w")
        output_file.write("Id,Category\n")
        torch.no_grad()
        for f in tqdm(os.listdir(self.config.test_dir)):
            if 'jpg' in f:
                data = self.data_loader.transform["val"](
                    pil_loader(self.config.test_dir + '/' + f))
                data = data.view(1, data.size(0), data.size(1), data.size(2))
                if self.cuda:
                    data = data.cuda()
                output = self.model.net(data)
                logit = self.loss(output, None)
                pred = torch.argmax(logit)
                output_file.write("%s,%d\n" % (f[:-4], pred.data.cpu()))
        output_file.close()

        print("Succesfully wrote " + self.config.outfile +
              ', you can upload this file to the kaggle competition website')

    def plot_sample_images(self):
        """Plot sample images afteer transform to visualize what is fed in the network
        """
        import matplotlib.pyplot as plt

        def denormalise(image):
            image = image.numpy().transpose(1, 2, 0)  # PIL images have channel last
            mean = [0.485, 0.456, 0.406]
            stdd = [0.229, 0.224, 0.225]
            image = (image * stdd + mean).clip(0, 1)
            return image

        example_rows = 2
        example_cols = 5
        # Get a batch of images and labels
        sampler = torch.utils.data.DataLoader(
            self.data_loader.train_dataset, batch_size=example_rows*example_cols, shuffle=True, num_workers=self.config.num_workers)
        images, indices = next(iter(sampler))
        plt.ioff()
        plt.rcParams['figure.dpi'] = 120  # Increase size of pyplot plots

        # Show a grid of example images
        # sharex=True, sharey=True)
        fig, axes = plt.subplots(example_rows, example_cols, figsize=(9, 5))
        axes = axes.flatten()
        labels = self.data_loader.train_dataset.classes
        for ax, image, index in zip(axes, images, indices):
            ax.imshow(denormalise(image))
            ax.set_axis_off()
            ax.set_title(labels[index.data], fontsize=7)

        fig.subplots_adjust(wspace=0.02, hspace=0)
        fig.suptitle('Augmented training set images', fontsize=20)
        wandb.log({"Random sample of transformed images": plt})
        plt.close()

    def getActivation(self,name):
    # the hook signature
        def hook(model, input, output):
            to_concatenate = torch.nn.functional.normalize(output.detach().cpu()).numpy().squeeze()
            if len(self.activation.shape) == 1 : self.activation = to_concatenate
            else: self.activation = np.append(self.activation,to_concatenate,axis = 0) 

        return hook

        
    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        print("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
