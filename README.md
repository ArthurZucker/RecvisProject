# Recvis final project : 

## Abstract 

Self-supervised learning using transformers has shown interesting emerging properties and learn rich embeddings without annotations. Most recently, Barlow Twins proposed an elegant self-supervised learning technique using a ResNet50 backbone which achieved competitive results when fine-tuned on downstream tasks. The network learns deep image embeddings based on a cross correlation loss which pushes a similarity between the embeddings of two crops of the same image. We propose to study a transformer based Barlow Twins architecture, while analyzing the learned embeddings and testing our method on semantic segmentation tasks. Our goal is to check the reputability of the relevant attention maps produced by DINO using the self supervised training method proposed in Barlow Twins. Then, we want to use such attention maps (either from DINO or our experiment) in order to assist the segmentation and create a new attention aware decoding architecture. 


## Project proposal 


# TO DO : 

- [x] Define metrics and baseline to compare our architectures 
- [x] Download dataset and create dataloader 
- [x] Visualize effective receptive field
- [ ] Create Transformer network
- [x] Create BarlowTwins agent 
- [ ] BoTnet
- [ ] SwinTransformer
- [ ] Create find biggest image size


1. Please clearly define quantitative measures and baseline methods with which you are going to compare results.
2. Please clearly define steps of your project from simple to more complicated and how you are going to evaluate each step (point 1. above). It is important to have some quantitative evaluation and comparison with baselines early on in the project so that you don’t run out of time. Having quantitative experiments and comparison with baselines (ideally published methods) is an important component of the final project evaluation.

Here no baseline is required, this is mostly a visual evaluation but we could invent a metric of how significant the attention heads are, how much information they carry, and thus could create a loss to push the attention heads to learn even more meaningful attentino maps. 


### 2. Semantic Segmentation on COCOStuff

Our goal is to design a new semantic segmentation head which uses the attention maps. We could look for inspiration from the most recent **SegFormer** paper which implements

We will use the following evaluations  :
- Barlow twins with resnet 50 + semantic segmentation head without attention maps 
- Barlow twins with SwinViT + semantic segmentation head, whithout using attention maps
- Barlow twins with SwinViT + semantic segmentation head, whith attention maps, fusion 1
- Barlow twins with SwinViT + semantic segmentation head, whith attention maps, fusion 2
- DINO weights + semantic segmentation head, whith attention maps, fusion 2

We have to find various ways of combining the different attention maps and heads and use them in the semantic segmentation head. 

## Planning and workload : 

### Tasks : 

- [x] Download dataset
- [ ] Take a look at google cloud platform and put the dataset inside
- [x] Use pytorch lightning
- [x] Implement Barlow Twins with resnet50 
- [ ] Implement Dino using hugging face transformers (?) / pytorch lighting? 
- [ ] BT SwinViT w/o attention
- [ ] BT SwinViT w attention
- [ ] Ablation studies
- [x] Feature visualization, embeddings etc

# References and links : 
https://github.com/bytedance/ibot/blob/main/analysis/attention_map/visualize_attention.py 

https://github.com/facebookresearch/barlowtwins



# Code to add : 

- [ ] Add AMP level : `trainer = Trainer(amp_level='O2')`
- [ ] Add SWA : `stochastic_weight_avg=True`

In vision transformer timm return all tokens in the forward (not only the cls token) (for segmentation)