# Recvis final project : 

## Abstract 

Self-supervised learning using transformers \cite{dosovitskiy2021image} has shown interesting emerging properties \cite{caron2021emerging} and learn rich embeddings without annotations. Most recently, Barlow Twins \cite{DBLP:journals/corr/abs-2103-03230} proposed an elegant self-supervised learning technique using a ResNet50 backbone which achieved competitive results when fine-tuned on downstream tasks. The network learns deep image embeddings based on a cross correlation loss which pushes a similarity between the embeddings of two crops of the same image. We propose to study a transformer based Barlow Twins architecture, while analyzing the learned embeddings and testing our method on semantic segmentation tasks. Our goal is to check the reputability of the relevant attention maps produced by \cite{caron2021emerging} using the self supervised training method proposed in Barlow Twins. Then, we want to use such attention maps (either from DINO or our experiment) in order to assist the segmentation and create a new attention aware decoding architecture. 


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

