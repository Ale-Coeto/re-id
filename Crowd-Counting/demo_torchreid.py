import os
# import torch.nn as nn
# import torch
import queue
import matplotlib.pyplot as plt
# from ReId_baseline_pytorch.reid_model import get_cosine_score
from extract_emb_torchreid import extract_features, get_cosine_score
from PIL import Image

import random

model_name = "mobilenetv2_x1_0r"
query_index = 47

# Features from a folder
def get_features(folder):
    features = []
    for filename in (os.listdir(folder)):
        path = os.path.join(folder, filename)
        image = Image.open(path)
        feature = extract_features(image)
        features.append({"f": feature, "file":path})

    return features

def plot(results, query_image, id):

    fig = plt.figure(figsize=(16,4))
    ax = plt.subplot(1,11,1)
    ax.axis('off')
    im = plt.imread(query_image["file"])
    plt.imshow(im)
    plt.title("Query Img", color='green')

    for i in range(10):
        if not results.empty():
            ax = plt.subplot(1,11,i+2)
            ax.axis('off')

            priority, file = results.get()
            im = plt.imread(file)
            plt.imshow(im)
            plt.title(round(-1*priority, 4))

    fig.savefig(os.path.join("./results", f"{model_name}_{id}"))


def query(features, id):
    if features == None:
        return
    query_image_index = id # random.randint(0, len(features)-1)
    print(query_image_index)
    
    query_image = features[query_image_index]
    
    print("q", query_image["file"])
    query_image_feature = query_image["f"]
    results = queue.PriorityQueue()

    for i,feature in enumerate(features):
        if i == query_image_index:
            continue
        cosine_distance = get_cosine_score(query_image_feature, feature["f"])
        
        results.put((cosine_distance, feature["file"]))
    
    plot(results, query_image, id)

    for _ in range(10):
        if not results.empty():
            priority, file = results.get()
            print(file)

    

def process(folder, id):
    features = get_features(folder)
    query(features, id) 


folder1 = "./cam1_cropped"
folder2 = "./cam2_cropped"
folder = "./test_cropped"

process(folder, query_index)






