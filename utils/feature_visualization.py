import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import wandb
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def get_representation(test_embeddings, test_predictions, val_embed,val_pred,dimension=3):
    """[summary]

    Args:
        test_embeddings np.array([float]): should contain the deep embeddings. Can be obtained using hooks
        test_predictions np.array([float]): contains the predicted output (from softmax)
        dimension (int, optional): Projection dimension to use for the TSNE algorithm. Defaults to 3.
    """
    # sns.set(style = "darkgrid")
    tsne = TSNE(dimension)
    X = np.append(test_embeddings,val_embed,axis=0)
    y = np.append(test_predictions,val_pred)
    tsne_proj = tsne.fit_transform(X)
    df_subset = pd.DataFrame(X)
    df_subset['y'] = y
    df_subset['label'] = df_subset['y'].apply(lambda i: str(i))
    df_subset['style'] = [["val","train"][i<len(test_predictions)] for i in range(len(y))] 
    # Plot those points as a scatter plot and label them based on the pred labels
    df_subset['tsne-3d-one'] = tsne_proj[:,0]
    df_subset['tsne-3d-two'] = tsne_proj[:,1]
    
    if dimension ==3 :
        df_subset['tsne-3d-three'] = tsne_proj[:,2]
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-3d-one", y="tsne-3d-two",
        hue="y",
        style = "style",
        data=df_subset,
        legend="full",
        palette="deep",
        alpha=0.75
    )
    plot = wandb.Image(plt)
    wandb.log({"Learned embedding":plot})
    plt.close()

def plot_contrastive(test_embeddings, test_predictions, val_embed,val_pred):

    fig = plt.figure(figsize=(16,10))
    ax = fig.add_subplot(111, projection='3d')

    embeds = np.append(test_embeddings,val_embed,axis=0)
    labels = np.append(test_predictions,val_pred)
    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)
    ax.plot_surface(
        x, y, z,  rstride=1, cstride=1, color='w', alpha=0.3, linewidth=0)
    ax.scatter(embeds[:,0], embeds[:,1], embeds[:,2], c=labels, s=20)

    
    # ax.set_xlim([-1, 1])
    # ax.set_ylim([-1, 1])
    # ax.set_zlim([-1, 1])
    
    ax.set_xlim([np.amin(x), np.amax(x)])
    ax.set_ylim([np.amin(y), np.amax(y)])
    ax.set_zlim([np.amin(z), np.amax(z)])
    
    # ax.set_aspect("equal")
    ax.set_box_aspect((np.amax(x)-np.amin(x), np.amax(y)-np.amin(y),np.amax(z)-np.amin(z)))
    plt.tight_layout()
    plot = wandb.Image(plt)
    wandb.log({"Sphere embedding":plot})
    plt.close()