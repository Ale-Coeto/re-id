import os
from ReId_baseline_pytorch.reid_model import load_network, get_structure, extract_feature_from_path, get_cosine_score
import torch.nn as nn
import torch
import queue
import matplotlib.pyplot as plt

# Load the ReID model
structure = get_structure()
model_reid = load_network(structure)
model_reid.classifier.classifier = nn.Sequential()
model_name = "swin"
query_index = 47

use_gpu = torch.cuda.is_available()
if use_gpu:
    model_reid = model_reid.cuda()

# Features from a folder
def get_features(folder):
    features = []
    for filename in (os.listdir(folder)):
        path = os.path.join(folder, filename)
        # print("Extracting", path)
        feature = extract_feature_from_path(path, model_reid)
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
    query_image = features[id]
    print("q", query_image["file"])
    query_image_feature = query_image["f"]
    results = queue.PriorityQueue()

    for i,feature in enumerate(features):
        if i == id:
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






