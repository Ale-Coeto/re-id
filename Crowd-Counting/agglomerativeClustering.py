import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))

from REID_baseline_pytorch.reid_model import load_network, compare_images, extract_feature_from_img, get_structure
from REID_baseline_pytorch.extract_emb_torchreid import extract_features_torchreid
# from ultralytics import YOLO
# import torch.nn as nn
# from PIL import Image
import numpy as np
# import torch
import cv2
import os
# from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import pickle

'''
Script to extract embeddings from people tracked in a video
'''


embeddings = np.load("embeddings/Swin.npy",allow_pickle=True)

print("len:", len(embeddings))
print("len emb:", len(embeddings[0]))

with open('./results/images.pickle', 'rb') as f:
    imgs = pickle.load(f)

clustering = AgglomerativeClustering(distance_threshold=25,n_clusters=None).fit(embeddings)

print("Clustering_labels:", clustering.labels_)

print("Number of clusters:", clustering.n_clusters_)

for i in range(clustering.n_clusters_):
    cur_imgs = []
    for j in range(len(clustering.labels_)):
        if (clustering.labels_[j] == i):
            cur_imgs.append(imgs[j])
    fig, axes = plt.subplots(1, len(cur_imgs), figsize=(len(cur_imgs) * 3, 5))
    
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    for i in range(len(cur_imgs)):
        axes[i].imshow(cur_imgs[i], cmap="gray")  # Assuming grayscale images
        axes[i].axis('off')  # Hide axes
    
    plt.tight_layout()
    plt.savefig(f'./results/clustering/agglomerative_{i}.png')
