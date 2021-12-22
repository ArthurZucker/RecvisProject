# Recvis final project : 

## Abstract 

Self-supervised learning using transformers \cite{dosovitskiy2021image} has shown interesting emerging properties \cite{caron2021emerging} and learn rich embeddings without annotations. Most recently, Barlow Twins \cite{DBLP:journals/corr/abs-2103-03230} proposed an elegant self-supervised learning technique using a ResNet50 backbone which achieved competitive results when fine-tuned on downstream tasks. The network learns deep image embeddings based on a cross correlation loss which pushes a similarity between the embeddings of two crops of the same image. We propose to study a transformer based Barlow Twins architecture, while analyzing the learned embeddings and testing our method on semantic segmentation tasks. Our goal is to check the reputability of the relevant attention maps produced by \cite{caron2021emerging} using the self supervised training method proposed in Barlow Twins. Then, we want to use such attention maps (either from DINO or our experiment) in order to assist the segmentation and create a new attention aware decoding architecture. 


## Project proposal 


# TO DO : 

- [ ] Define metric and baseline to compare our architectures 
- [ ] Download dataset and create dataloader 
- [ ] Create Transformer network
- [ ] Create BarlowTwins agent 
- [ ] Implement feature map visualization 

## Defining baseline : 

### 1. Emerging properties of self supervise learning and attention 

Here no baseline is required, this is mostly a visual evaluation but we could invent a metric of how significant the attention heads are, how much information they carry, and thus could create a loss to push the attention heads to learn even more meaningful attentino maps. 


### 2. Semantic Segmentation on COCOStuff

Our goal is to design a new semantic segmentation head which uses the attention maps. We could look for inspiration from the most recent **SegFormer** paper which implements

We will use the following evaluations  :
- Barlow twins with resnet 50 + semantic segmentation head without attention maps 
- Barlow twins with SwinViT + semantic segmentation head, whithout using attention maps
-  Barlow twins with SwinViT + semantic segmentation head, whith attention maps, fusion 1
-  Barlow twins with SwinViT + semantic segmentation head, whith attention maps, fusion 2
-  DINO weights + semantic segmentation head, whith attention maps, fusion 2

We have to find various ways of combining the different attention maps and heads and use them in the semantic segmentation head. 

## Planning and workload : 

### Tasks : 

- [ ] Download dataset
- [ ] Take a look at google cloud platform and put the dataset inside
- [ ] Use pytorch lightning
- [ ] Implement Barlow Twins with resnet50 
- [ ] Implement Dino using hugging face transformers (?) / pytorch lighting? 
- [ ] BT SwinViT w/o attention
- [ ] BT SwinViT w attention
- [ ] Ablation studies
- [ ] Feature visualization, embeddings etc

# References and links : 
https://github.com/bytedance/ibot/blob/main/analysis/attention_map/visualize_attention.py 

https://github.com/facebookresearch/barlowtwins

# wandb key : 
1bd5c9f2298e5875d25866099bd98a8437c50cb6 


# Code to add : 

- [ ] Add AMP level : `trainer = Trainer(amp_level='O2')`
- [ ] Add SWA : `stochastic_weight_avg=True`

```
📦RecvisProject
 ┣ 📂.vscode
 ┃ ┗ 📜launch.json
 ┣ 📂agents
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┣ 📜__init__.cpython-38.pyc
 ┃ ┃ ┣ 📜base.cpython-38.pyc
 ┃ ┃ ┣ 📜base_contrastive.cpython-38.pyc
 ┃ ┃ ┗ 📜trainer.cpython-38.pyc
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜base.py
 ┃ ┣ 📜base_contrastive.py
 ┃ ┗ 📜trainer.py
 ┣ 📂assets
 ┃ ┣ 📂VOC
 ┃ ┃ ┣ 📂VOCdevkit
 ┃ ┃ ┃ ┗ 📂VOC2012
 ┃ ┃ ┃ ┃ ┣ 📂Annotations
 ┃ ┃ ┃ ┃ ┃ ┣ 📜2007_000027.xml
 ┃ ┃ ┃ ┃ ┃ ┣ 📜2007_000032.xml
 ┃ ┃ ┃ ┃ ┣ 📂ImageSets
 ┃ ┃ ┃ ┃ ┃ ┣ 📂Action
 ┃ ┃ ┃ ┃ ┃ ┃ ┣ 📜jumping_train.txt
 ┃ ┃ ┃ ┃ ┃ ┣ 📂Layout
 ┃ ┃ ┃ ┃ ┃ ┃ ┣ 📜train.txt
 ┃ ┃ ┃ ┃ ┃ ┃ ┣ 📜trainval.txt
 ┃ ┃ ┃ ┃ ┃ ┃ ┗ 📜val.txt
 ┃ ┃ ┃ ┃ ┃ ┣ 📂Main
 ┃ ┃ ┃ ┃ ┃ ┃ ┣ 📜aeroplane_train.
 ┃ ┃ ┃ ┃ ┃ ┗ 📂Segmentation
 ┃ ┃ ┃ ┃ ┃ ┃ ┣ 📜train.txt
 ┃ ┃ ┃ ┃ ┃ ┃ ┣ 📜trainval.txt
 ┃ ┃ ┃ ┃ ┃ ┃ ┗ 📜val.txt
 ┃ ┃ ┃ ┃ ┣ 📂JPEGImages
 ┃ ┃ ┃ ┃ ┃ ┣ 📜2007_000027.jpg
 ┃ ┃ ┃ ┃ ┃ ┣ 📜2007_000032.jpg
 ┃ ┃ ┃ ┃ ┣ 📂SegmentationClass
 ┃ ┃ ┃ ┃ ┃ ┣ 📜2007_000032.png
 ┃ ┃ ┃ ┃ ┃ ┣ 📜2007_000033.png
 ┃ ┃ ┃ ┃ ┗ 📂SegmentationObject
 ┃ ┃ ┃ ┃ ┃ ┣ 📜2007_000032.png
 ┃ ┃ ┃ ┃ ┃ ┣ 📜2007_000033.png
 ┃ ┃ ┗ 📜VOCtrainval_11-May-2012.tar
 ┃ ┗ 📜.keep
 ┣ 📂config
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┗ 📜hparams.cpython-38.pyc
 ┃ ┗ 📜hparams.py
 ┣ 📂datamodule
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┣ 📜VOCSegmentationDataModule.cpython-38.pyc
 ┃ ┃ ┣ 📜VOCsegmentationmodule.cpython-38.pyc
 ┃ ┃ ┣ 📜__init__.cpython-38.pyc
 ┃ ┃ ┗ 📜base_data_module.cpython-38.pyc
 ┃ ┣ 📜VOCSegmentationDataModule.py
 ┃ ┣ 📜__init__.py
 ┃ ┗ 📜base_data_module.py
 ┣ 📂datasets
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┣ 📜BirdsDataloader.cpython-38.pyc
 ┃ ┃ ┣ 📜__init__.cpython-38.pyc
 ┃ ┃ ┣ 📜base_dataloader.cpython-38.pyc
 ┃ ┃ ┗ 📜base_dataset.cpython-38.pyc
 ┃ ┣ 📜BirdsDataloader.py
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜base_dataloader.py
 ┃ ┗ 📜base_dataset.py
 ┣ 📂graphs
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┣ 📜__init__.cpython-38.pyc
 ┃ ┃ ┗ 📜weights_initializer.cpython-38.pyc
 ┃ ┣ 📂losses
 ┃ ┃ ┣ 📂__pycache__
 ┃ ┃ ┃ ┣ 📜Angular.cpython-38.pyc
 ┃ ┃ ┃ ┣ 📜Subcenter_arcface.cpython-38.pyc
 ┃ ┃ ┃ ┣ 📜__init__.cpython-38.pyc
 ┃ ┃ ┃ ┗ 📜custom_loss.cpython-38.pyc
 ┃ ┃ ┣ 📜Angular.py
 ┃ ┃ ┣ 📜Subcenter_arcface.py
 ┃ ┃ ┣ 📜__init__.py
 ┃ ┃ ┗ 📜custom_loss.py
 ┃ ┣ 📂models
 ┃ ┃ ┣ 📂__pycache__
 ┃ ┃ ┃ ┣ 📜Contrastive_resnet50.cpython-38.pyc
 ┃ ┃ ┃ ┣ 📜Contrastive_vit.cpython-38.pyc
 ┃ ┃ ┃ ┣ 📜__init__.cpython-38.pyc
 ┃ ┃ ┃ ┣ 📜base.cpython-38.pyc
 ┃ ┃ ┃ ┣ 📜mobileNet.cpython-38.pyc
 ┃ ┃ ┃ ┣ 📜resnet50.cpython-38.pyc
 ┃ ┃ ┃ ┗ 📜vit.cpython-38.pyc
 ┃ ┃ ┣ 📂custom_layers
 ┃ ┃ ┃ ┣ 📂__pycache__
 ┃ ┃ ┃ ┃ ┣ 📜__init__.cpython-38.pyc
 ┃ ┃ ┃ ┃ ┗ 📜layer_norm.cpython-38.pyc
 ┃ ┃ ┃ ┣ 📜__init__.py
 ┃ ┃ ┃ ┗ 📜layer_norm.py
 ┃ ┃ ┣ 📜Contrastive_resnet50.py
 ┃ ┃ ┣ 📜Contrastive_vit.py
 ┃ ┃ ┣ 📜__init__.py
 ┃ ┃ ┣ 📜base.py
 ┃ ┃ ┣ 📜mobileNet.py
 ┃ ┃ ┣ 📜resnet50.py
 ┃ ┃ ┗ 📜vit.py
 ┃ ┣ 📜.DS_Store
 ┃ ┣ 📜__init__.py
 ┃ ┗ 📜weights_initializer.py
 ┣ 📂model
 ┃ ┗ 📜base_voc.py
 ┣ 📂models
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┣ 📜__init__.cpython-38.pyc
 ┃ ┃ ┣ 📜base.cpython-38.pyc
 ┃ ┃ ┗ 📜base_voc.cpython-38.pyc
 ┃ ┣ 📂custom_layers
 ┃ ┃ ┣ 📂__pycache__
 ┃ ┃ ┃ ┣ 📜__init__.cpython-38.pyc
 ┃ ┃ ┃ ┣ 📜layer_norm.cpython-38.pyc
 ┃ ┃ ┃ ┗ 📜unet_convs.cpython-38.pyc
 ┃ ┃ ┣ 📜__init__.py
 ┃ ┃ ┣ 📜layer_norm.py
 ┃ ┃ ┗ 📜unet_convs.py
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜base.py
 ┃ ┗ 📜base_voc.py
 ┣ 📂scripts
 ┃ ┗ 📜sweep.yml
 ┣ 📂test-sem-seg
 ┃ ┗ 📂jdv4pql6
 ┃ ┃ ┗ 📂checkpoints
 ┣ 📂utils
 ┃ ┣ 📂__pycache__
 ┃ ┃ ┣ 📜__init__.cpython-38.pyc
 ┃ ┃ ┣ 📜agent_utils.cpython-38.pyc
 ┃ ┃ ┣ 📜callbacks.cpython-38.pyc
 ┃ ┃ ┣ 📜feature_visualization.cpython-38.pyc
 ┃ ┃ ┣ 📜logger.cpython-38.pyc
 ┃ ┃ ┣ 📜metrics.cpython-38.pyc
 ┃ ┃ ┣ 📜misc.cpython-38.pyc
 ┃ ┃ ┗ 📜transforms.cpython-38.pyc
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜agent_utils.py
 ┃ ┣ 📜callbacks.py
 ┃ ┣ 📜feature_visualization.py
 ┃ ┣ 📜logger.py
 ┃ ┣ 📜metrics.py
 ┃ ┣ 📜misc.py
 ┃ ┗ 📜transforms.py
 ┣ 📂wandb
 ┃ ┣ 📂latest-run
 ┃ ┃ ┣ 📂files
 ┃ ┃ ┃ ┣ 📜conda-environment.yaml
 ┃ ┃ ┃ ┣ 📜config.yaml
 ┃ ┃ ┃ ┣ 📜output.log
 ┃ ┃ ┃ ┣ 📜requirements.txt
 ┃ ┃ ┃ ┣ 📜wandb-metadata.json
 ┃ ┃ ┃ ┗ 📜wandb-summary.json
 ┃ ┃ ┣ 📂logs
 ┃ ┃ ┃ ┣ 📜debug-internal.log
 ┃ ┃ ┃ ┗ 📜debug.log
 ┃ ┃ ┣ 📂tmp
 ┃ ┃ ┃ ┗ 📂code
 ┃ ┃ ┗ 📜run-1oueugp8.wandb
 ┃ ┣ 📂run-20211220_180546-16kueqtx
 ┃ ┃ ┣ 📂files
 ┃ ┃ ┃ ┣ 📜conda-environment.yaml
 ┃ ┃ ┃ ┣ 📜config.yaml
 ┃ ┃ ┃ ┣ 📜output.log
 ┃ ┃ ┃ ┣ 📜requirements.txt
 ┃ ┃ ┃ ┣ 📜wandb-metadata.json
 ┃ ┃ ┃ ┗ 📜wandb-summary.json
 ┃ ┃ ┣ 📂logs
 ┃ ┃ ┃ ┣ 📜debug-internal.log
 ┃ ┃ ┃ ┗ 📜debug.log
 ┃ ┃ ┣ 📂tmp
 ┃ ┃ ┃ ┗ 📂code
 ┃ ┃ ┗ 📜run-16kueqtx.wandb 
 ┣ 📂weights
 ┃ ┗ 📜.keep
 ┣ 📜.git
 ┣ 📜.gitignore
 ┣ 📜LICENSE
 ┣ 📜README.md
 ┣ 📜__init__.py
 ┣ 📜main.py
 ┣ 📜requirements.txt
 ┗ 📜run.sh
 ```
 test