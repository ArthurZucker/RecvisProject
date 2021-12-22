import wandb
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
import numpy as np
import torch
import PIL

from utils.transforms import UnNormalize

PASCAL_VOC_classes = {
    0: "background",
    1: "airplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "table",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "potted_plant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tv",
    21: "void",
}

class LogPredictionsCallback(Callback):
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        """Called when the validation batch ends."""

        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case

        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0:
            n = 20
            x, y = batch
            images = x[:n].cpu()
            ground_truth = np.array(y[:n].cpu())
            predictions = np.array(outputs[:n].cpu())
            # captions = [
            #     f"Ground Truth: {y_i} - Prediction: {y_pred}"
            #     for y_i, y_pred in zip(y[:n], outputs[:n])
            # ]n
            samples = []
            mean = [0.485, 0.456, 0.406] # TODO this is not beautiful
            std = [0.229, 0.224, 0.225]
            UnNormalizer = UnNormalize(mean, std)
            for i in range(len(batch)):
                bg_image = np.clip(images[i],0,1).numpy()
                bg_image = np.transpose(bg_image, (1, 2, 0))
                # run the model on that image
                prediction_mask = predictions[i] #FIXME masks are not correct
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
            wandb.log({"predictions":samples})
            # trainer.logger.log_image(
            # key="sample_images",
            # samples=samples,
            # )

        # TODO add sample input image visualization



class LogFeatureVisualizationCallback(Callback):
    pass
    # TODO add feature vizualization for training and validation data
    # should probably use hooks to keep the model structure