# Study of the Emerging Properties of Self-Supervised Vision Transformers and Semantic Segmentation  : 

## Abstract 

Self-supervised learning using transformers has shown interesting emerging properties and learn rich embeddings without annotations. Most recently, Barlow Twins proposed an elegant self-supervised learning technique using a ResNet-50 backbone which achieved competitive results when fine-tuned on downstream tasks. In this paper, we propose to study Vision Transformers trained using the Barlow Twins self-supervised method, and compare the results with. We demonstrate the effectiveness of the Barlow Twins method by showing that networks pretrained on the small PASCAL VOC 2012 dataset are able to generalize well while requiring less training and computing power than the DINO method. Finally we propose to leverage self-supervised vision transformers and their semantically rich attention maps for semantic segmentation tasks.
## Project report 

You can find the complete project report in the `assets/pdf` repository or click [here](assets/pdf/FPR_Apavou_Zucker.pdf). Our slides are also available [here](https://docs.google.com/presentation/d/1MvE78E8pb4XEIMQxZLkBZnLMXvVNC8E-m7rnxW5wa8Q/edit?usp=sharing).

# Setting up the envirronement 

We exported the required packages in a `requirement.txt` file taht can be used as follows : 
```
pip install -r requirements.txt
```

# Training 

Refer to the [Barlow Twins Wiki]( ) and the [Semantic Segmentation Wiki ]() for more details

# Contributions : 
https://github.com/facebookresearch/barlowtwins

# Visualiation 
We implemented two very efficient and easy-to use callbacks to visualize the effective receptive fields and the attention maps at train and validation time. 
Examples are shown in [1] and [2]. Both rely on pytorch `hooks` and provide more interpretation to the training. Both were implemented from scratch, and the visualization of the effective receptive fields is based on the theory from \cite{luo2017understanding}.

We also logged the evolution of the cross-correlation matrix which is fare more interpretable than the value of the loss. As various training showed, a decreasing loss can have a cross-correlation matrix far from the identity. We used a heatmap to represent the empirical cross correlation matrix were values close to 1 are red and values close to zeros are cyan blue. An example of the cross-correlation matrix of a fully converged model can be found in [3].

# Acknowledgments: 

Our implementation relies on `pytorch lightning`, and thus requires its installation. We also use the `rich` library for nicer progress bars and the very handy `wandb` library to visualize our experiments.  

## Semantic Segmentation on COCOStuff

Our goal is to design a new semantic segmentation head which uses the attention maps. We could look for inspiration from the most recent **SegFormer** paper which implements

We will use the following evaluations  :
- Barlow twins with resnet 50 + semantic segmentation head without attention maps 
- Barlow twins with SwinViT + semantic segmentation head, whithout using attention maps
- Barlow twins with SwinViT + semantic segmentation head, whith attention maps, fusion 1
- Barlow twins with SwinViT + semantic segmentation head, whith attention maps, fusion 2
- DINO weights + semantic segmentation head, whith attention maps, fusion 2

We have to find various ways of combining the different attention maps and heads and use them in the semantic segmentation head. 
# References and links : 
https://github.com/bytedance/ibot/blob/main/analysis/attention_map/visualize_attention.py 



In `model/fix_tim/vision_transformer` the vision transformer returns every token in the forward pass(while only the cls token is usually returned) (for segmentation)
