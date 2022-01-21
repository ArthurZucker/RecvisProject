import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torchvision
import wandb
from pytorch_lightning.callbacks import Callback
from torch.autograd import grad

from utils.constant import PASCAL_VOC_classes
from utils.hooks import get_attention, get_activation
from utils.metrics_module import MetricsModule

class LogTransformedImages(Callback):
    def __init__(self,log_img_freq) -> None:
        super().__init__()
        self.log_img_freq = log_img_freq
    
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0 and pl_module.current_epoch % self.log_img_freq == 0:
            self.log_images("validation", batch, 5, outputs)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the training batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0 and pl_module.current_epoch % self.log_img_freq == 0:
            self.log_images("train", batch, 5, outputs)

    def log_images(self, name, batch, n, outputs):

        x, y = batch
        images = x[:n].cpu()
        ground_truth = np.array(y[:n].cpu())

        samples = []

        mean = np.array([0.485, 0.456, 0.406])  # TODO this is not beautiful
        std = np.array([0.229, 0.224, 0.225])

        for i in range(images.shape[0]):

            bg_image = images[i].detach().numpy().transpose((1, 2, 0))
            bg_image = std * bg_image + mean
            bg_image = np.clip(bg_image, 0, 1)

            samples.append(wandb.Image(bg_image,))
            
        wandb.log({f"transformed images/{name}": samples})
        

class LogSegmentationCallback(Callback):
    def __init__(self, log_img_freq) -> None:
        super().__init__()
        self.log_img_freq = log_img_freq
        
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0 and pl_module.current_epoch % self.log_img_freq == 0:
            self.log_images("validation", batch, 5, outputs)

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the training batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0 and pl_module.current_epoch % self.log_img_freq == 0:
            self.log_images("train", batch, 5, outputs)

    def log_images(self, name, batch, n, outputs):

        x, y = batch
        images = x[:n].cpu()
        ground_truth = np.array(y[:n].cpu())

        logits = outputs["logits"]  # preds
        preds = torch.argmax(logits, dim=1)

        predictions = np.array(preds[:n].cpu())

        samples = []

        mean = np.array([0.485, 0.456, 0.406])  # TODO this is not beautiful
        std = np.array([0.229, 0.224, 0.225])

        for i in range(images.shape[0]):

            bg_image = images[i].detach().numpy().transpose((1, 2, 0))
            bg_image = std * bg_image + mean
            bg_image = np.clip(bg_image, 0, 1)

            prediction_mask = predictions[i]
            true_mask = ground_truth[i]

            samples.append(
                wandb.Image(
                    bg_image,
                    masks={
                        "prediction": {
                            "mask_data": prediction_mask,
                            "class_labels": PASCAL_VOC_classes,
                        },
                        "ground truth": {
                            "mask_data": true_mask,
                            "class_labels": PASCAL_VOC_classes,
                        },
                    },
                )
            )
        wandb.log({name: samples})

class LogMetricsCallback(Callback):
    def __init__(self, config):
        self.config = config

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        device = pl_module.device

        self.metrics_module_train = MetricsModule(
            "train", self.config, device
        )

        self.metrics_module_validation = MetricsModule(
            "val", self.config, device
        )

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the train batch ends."""

        _, y = batch
        self.metrics_module_train.update_metrics(outputs["logits"], y)

    def on_train_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""

        self.metrics_module_train.log_metrics("train/", pl_module)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""

        _, y = batch
        self.metrics_module_validation.update_metrics(outputs["logits"], y)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the validation epoch ends."""

        self.metrics_module_validation.log_metrics("val/", pl_module)


class LogBarlowCCMatrixCallback(Callback):
    """Logs the cross correlation matrix obtain
    when computing the loss. This gives us an idea of
    how the network learns.
    TODO : when should we log ?
    TODO : should we average over batches only? Or epoch?
    For now, the average over the epoch will be computed
    as a moving average.
    A hook should be registered on the loss, using a new argument in the loss
    loss.cc_M which will be stored each time and then deleted

    """

    def __init__(self, log_ccM_freq) -> None:
        super().__init__()
        self.log_ccM_freq = log_ccM_freq
        self.cc_M = None

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the training batch ends."""
        # Let's log 20 sample image predictions from first batch
        if self.cc_M is not None:
            self.cc_M += (pl_module.loss.cc_M - self.cc_M) / (batch_idx + 1)
        else:
            self.cc_M = pl_module.loss.cc_M
        del pl_module.loss.cc_M

        if batch_idx == 0 and pl_module.current_epoch % self.log_ccM_freq == 0:
            self.log_cc_M("train")

    def on_val_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the training batch ends."""
        # Let's log 20 sample image predictions from first batch
        if self.cc_M is not None:
            self.cc_M += (pl_module.loss.cc_M - self.cc_M) / (batch_idx + 1)
        else:
            self.cc_M = pl_module.loss.cc_M
        del pl_module.loss.cc_M

        if batch_idx == 0:
            self.log_cc_M("val")

    def log_cc_M(self, name):
        heatmap = self.cc_M
        ax = sns.heatmap(heatmap, cmap="rainbow", cbar=False)
        plt.title(f"Cross correlation matrix")
        ax.set_axis_off()
        ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        ax.margins(x=0,y=0)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
            hspace = 0, wspace = 0)
        wandb.log({f"cc_Matrix/{name}": (wandb.Image(plt))})
        plt.close()
        self.cc_M = None


class LogERFVisualizationCallback(Callback):
    # FIXME memory issue 
    def __init__(self, nb_erf,erf_freq,b_size) -> None:
        """Initialize the callback with the layers
        to use to compute the effective receptive fields
        FOr now TODO define the format (most probably ints for the stage or the index of the layer)

        Args:
            layers ([type]): [description]
        """
        super().__init__()
        self.erf_freq = erf_freq
        self.nb_erf = nb_erf
        self.eps = 1e-7
        self.gradient = {i: self.eps for i in range(self.nb_erf)}
        self.batch_size = b_size

    def on_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if (pl_module.current_epoch) % self.erf_freq == 0:
            self._register_layer_hooks(pl_module)
            pl_module.rq_grad = True

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        # hooks have already been placed to extract the output
        # here, compute the gradient of the various ouputs with respect
        # to the input feature map, average it over the validation batches
        # until we have the number of images we required
        if (
            not trainer.sanity_checking
            and pl_module.current_epoch % self.erf_freq == 0
        ):  # exclude last batch were batch norm is appliedss FIXME
            x, _ = batch
            for name in range(self.nb_erf):
                # for each layer, compute the mean
                gradient_wrt_ipt = self.features[name]
                if gradient_wrt_ipt != []:
                    try:
                        gradient_wrt_ipt = grad(gradient_wrt_ipt, x, retain_graph=True)[
                            0
                        ].detach()
                        # TODO check whether the abs should be done before the mean or after
                        self.gradient[name] += (
                            np.squeeze(
                                torch.mean(
                                    torch.sum(torch.abs(gradient_wrt_ipt), axis=1),
                                    axis=0,
                                )
                                .cpu()
                                .numpy()
                            )
                            - self.gradient[name]
                        ) / (batch_idx + 1)
                        # average over the batches but sum over the channels

                        del self.features[name]
                        self.features[name] = []  # reset the hooks for the batch
                    except Exception as e:
                        # the gradient can't be computed because of batchnorm, jsut ignore
                        print(f"Tried to compute gradient error : {e}, cleaning up")
                        del self.features[name]
                        self.features[name] = []  # reset the hooks for the batch

            if batch_idx % self.batch_size == 0:
                heatmaps = []
                for name in self.gradient:
                    heatmap = self.gradient[name]
                    # average the gradients over the batches but sum it over the channels
                    
                    if type(heatmap) != float:  # FIXE ME
                        plt.ioff()
                        heatmap = heatmap / (batch_idx + 1) * self.batch_size
                        heatmap = (
                            heatmap
                            - np.min(heatmap) / np.max(heatmap)
                            - np.min(heatmap)
                        )
                        ax = sns.heatmap(heatmap, cmap="rainbow", cbar=False)
                        plt.title(f"Layer {self.erf_layers_names[name]}")
                        ax.set_axis_off()
                        
                        heatmaps.append(wandb.Image(plt))
                        plt.close()

                if len(heatmaps) != 0:
                    wandb.log({f"heatmaps": heatmaps})

    def on_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:

        if (pl_module.current_epoch) % self.erf_freq == 0:

            self.gradient = {
                i: self.eps for i in range(self.nb_erf)
            }  # in case we wanna log on other epochs
            for hooks in self.hooks:
                hooks.remove()
            pl_module.rq_grad = False
            
    def _register_layer_hooks(self,pl_module): # @URGENT @TODO should be in the ERF Callabck function, would be cleaner
        self.hooks = []
        nb_erf = self.nb_erf #TODO only use those layers not every layer
        named_layers = list(pl_module.named_modules())[2:]
        if "loss" in named_layers[-1]:
            named_layers = named_layers[:-1]
        layer_span = ((len(named_layers))//nb_erf) # span to take each layers
        selected_layers = named_layers[::layer_span][:nb_erf-1]+[named_layers[-1]]
        self.erf_layers_names = list(dict(selected_layers).keys())
        if nb_erf >= 2:
            self.features = {idx:[] for idx in range(len(selected_layers))} # trick to always take first and last layers FIXME later
        for i,(name, module) in enumerate(selected_layers):
            #self.hooks.append(dict(dict_layers)[name].register_forward_hook(get_activation(i, self.features)))
            self.hooks.append(module.register_forward_hook(get_activation(i, self.features)))
        



class LogBarlowPredictionsCallback(Callback):
    def __init__(self,erf_freq) -> None:
        super().__init__()
        self.erf_freq = erf_freq
    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the training batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0 and pl_module.current_epoch % self.erf_freq == 0:
            self.log_images("train", batch, min(5,len(batch)), outputs)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the training batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0 and pl_module.current_epoch % self.erf_freq == 0:
            self.log_images("val", batch,  min(5,len(batch)), outputs)

    def log_images(self, name, batch, n, outputs):

        x1, x2 = batch
        image1 = x1[:n].cpu().detach().numpy()
        image2 = x2[:n].cpu().detach().numpy()

        samples1 = []
        samples2 = []
        mean = np.array([0.485, 0.456, 0.406])  # TODO this is not beautiful
        std = np.array([0.229, 0.224, 0.225])

        for i in range(n):

            bg1 = image1[i].transpose((1, 2, 0))
            bg1 = std * bg1 + mean
            bg1 = np.clip(bg1, 0, 1)

            bg2 = image2[i].transpose((1, 2, 0))
            bg2 = std * bg2 + mean
            bg2 = np.clip(bg2, 0, 1)

            samples1.append(wandb.Image(bg1))
            samples2.append(wandb.Image(bg2))

        wandb.log({f"{name}/x1": samples1})
        wandb.log({f"{name}/x2": samples2})  # TODO merge graphs


class LogAttentionMapsCallback(Callback):
    """Should only be used durng the fine-tuning task on a pretrained backbone
    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)
    """

    def __init__(self, attention_threshold, nb_attention, log_att_freq) -> None:
        super().__init__()
        self.log_freq = log_att_freq
        self.threshold = attention_threshold
        self.nb_attention_images = nb_attention

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ) -> None:
        if batch_idx == 0 and pl_module.current_epoch % self.log_freq == 0:
            self.hooks = []
            self.hooks.append(self._register_layer_hooks(pl_module))

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if batch_idx == 0 and pl_module.current_epoch % self.log_freq == 0:
            attention_maps = []
            th_attention_map = []
            for i in range(min(self.nb_attention_images,len(batch[0]))):
                img = batch[0][i]  
                # only 1 image for now. The batch has [0,1,...,n_1] crops b_size images
                w, h = (
                    img.shape[1] - img.shape[1] % pl_module.patch_size,
                    img.shape[2] - img.shape[2] % pl_module.patch_size,
                )
                img = img[:, :w, :h].unsqueeze(0)

                w_featmap = img.shape[-2] // pl_module.patch_size
                h_featmap = img.shape[-1] // pl_module.patch_size

                attentions = self.attention[0][i]
                # 0 is for the crop
                # i is for the image in the batch
                # extracts the attention maps for each head, corresponding to the first global crop, and the i-th image of the crop
                # attention are obtained from hooks

                nh = attentions.shape[0]  # number of head

                attentions = torch.tensor(attentions[:, 0, 1:].reshape(nh, -1))

                if self.threshold is not None:
                    # we keep only a certain percentage of the mass
                    val, idx = torch.sort(attentions)
                    val /= torch.sum(val, dim=1, keepdim=True)
                    cumval = torch.cumsum(val, dim=1)
                    th_attn = cumval > (1 - self.threshold)
                    idx2 = torch.argsort(idx)
                    for head in range(nh):
                        th_attn[head] = th_attn[head][idx2[head]]
                    th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
                    # interpolate
                    th_attn = nn.functional.interpolate(
                        th_attn.unsqueeze(0),
                        scale_factor=pl_module.patch_size,
                        mode="nearest",
                    )[0].cpu()
                    # lets now display the attentions thresholded for each heads on a single map

                    th_attention_map.append(th_attn)

                attentions = attentions.reshape(nh, w_featmap, h_featmap)
                attentions = nn.functional.interpolate(
                    attentions.unsqueeze(0),
                    scale_factor=pl_module.patch_size,
                    mode="nearest",
                )[0].cpu()

                plt.ioff()
                grid_img = torchvision.utils.make_grid(
                    attentions, normalize=True, scale_each=True, nrow=nh // 2
                )
                attention_maps.append(
                    [img.squeeze(0).cpu().numpy()] + list(grid_img.numpy())
                )
                del grid_img

            self.show(attention_maps, th_attention_map)

            del attention_maps,th_attention_map
            self._clear_hooks()


    def show(self, imgs, th_attention_map):
        import torchvision.transforms.functional as F

        plt.ioff()
        fix, axs = plt.subplots(nrows=len(imgs), ncols=len(imgs[0]) + 1, squeeze=True)
        mean = np.array([0.485, 0.456, 0.406])  # TODO this is not beautiful
        std = np.array([0.229, 0.224, 0.225])
        for j, sample in enumerate(imgs):
            for i, head in enumerate(sample):
                if i == 0:  # original crop
                    org = np.asarray(head).transpose(1, 2, 0)
                    axs[j, 0].imshow(np.clip(mean * org + std, 0, 1))
                else:
                    img = head
                    img = F.to_pil_image(img)
                    axs[j, i].imshow(np.asarray(img))
                axs[j, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            org = np.asarray(sample[0]).transpose(1, 2, 0)
            image = np.clip(mean * org + std, 0, 1)
            self.log_th_attention(
                image, th_attention_map[j], axs[j, i + 1]
            )  # log the thresholded attention maps
        fix.tight_layout()
        fix.subplots_adjust(left = 0, right = 1, top = 1, bottom = 0)
        fix.subplots_adjust(wspace=0.005, hspace=0.005)
        attention_heads = wandb.Image(plt)
        wandb.log({"attention heads": attention_heads})
        plt.close()
        del attention_heads

    def log_th_attention(self, image, th_att, ax):
        """th_attn should have every thrsholded attention maps for each heds, and each image that is being worked on"""
        import colorsys
        import random

        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from skimage.measure import find_contours

        def random_colors(N, bright=True):
            """
            Generate random colors.
            """
            brightness = 1.0 if bright else 0.7
            hsv = [(i / N, 1, brightness) for i in range(N)]
            colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
            random.shuffle(colors)
            return colors

        def apply_mask(image, mask, color, alpha=0.5):
            for c in range(3):
                image[:, :, c] = (
                    image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
                )
            return image

        N = th_att.shape[0]
        mask = th_att
        # Generate random colors
        colors = random_colors(N)
        masked_image = image.astype(np.uint32).copy()
        contour = True
        for i in range(N):
            color = colors[i]
            _mask = mask[i].numpy()
            # Mask
            masked_image = apply_mask(masked_image, _mask, color, alpha=0.5)
            ax.imshow(masked_image.astype(np.uint8))
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        del masked_image
        
        
    def _register_layer_hooks(self, pl_module):

        self.hooks = []
        named_layers = dict(pl_module.named_modules())
        attend_layers = []
        for name in named_layers:
            if ".attend" in name:
                attend_layers.append(named_layers[name])
        self.attention = []
        self.hooks.append(
            attend_layers[-1].register_forward_hook(get_attention(self.attention))
        )

    def _clear_hooks(self):
        for hk in self.hooks:
            hk.remove()
        del self.hooks
