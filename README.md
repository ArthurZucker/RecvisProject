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