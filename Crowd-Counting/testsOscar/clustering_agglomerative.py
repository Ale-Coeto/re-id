# from REID_baseline_pytorch.reid_model import load_network, compare_images, extract_feature_from_img, get_structure
# from REID_baseline_pytorch.extract_emb_torchreid import extract_features_torchreid
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
import collections

'''
Script to extract embeddings from people tracked in a video
'''


# Load inputs
out_dir = "./results"
in_dir_emb = os.path.join(out_dir, "embeddings")
in_path_images = os.path.join(out_dir, "images.pickle")
in_path = os.path.join(in_dir_emb, "SwinOsnet.pickle")

with open(in_path, 'rb') as f:
    embeddings_info = pickle.load(f)

with open(in_path_images, 'rb') as f:
    imgs = pickle.load(f)

out_dir_clustering = os.path.join(out_dir, "clustering")
os.makedirs(out_dir_clustering, exist_ok=True)

emb_img = {}
embeddings = []

for key in embeddings_info.keys():
    emb_img[len(embeddings)] = key # get index of current image
    embeddings.append(embeddings_info[key]["embedding"])


# clustering = AgglomerativeClustering(distance_threshold=50, n_clusters= None).fit(embeddings)
clustering = AgglomerativeClustering(n_clusters=7).fit(embeddings)

print("Clustering_labels:", clustering.labels_)

print("Number of clusters:", clustering.n_clusters_)

for i in range(clustering.n_clusters_):

    cur_imgs = []
    for j in range(len(clustering.labels_)):
        if (clustering.labels_[j] == i):
            cur_imgs.append(imgs[emb_img[j]])
    fig, axes = plt.subplots(1, len(cur_imgs), figsize=(len(cur_imgs) * 3, 5))
    
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    for j in range(len(cur_imgs)):
        axes[j].imshow(cur_imgs[j], cmap="gray")  # Assuming grayscale images
        axes[j].axis('off')  # Hide axes
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_clustering, f"agglomerative_{i}.png"))

