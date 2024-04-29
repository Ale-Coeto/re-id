import pickle
import numpy as np  
import os
import cv2

with open('./results/images.pickle', 'rb') as f:
    imgs = pickle.load(f)

for i, img in enumerate(imgs):
    cv2.imwrite(f'./results/features/img{i}.jpg', img)

embeddings = np.load("embeddings/Swin.npy",allow_pickle=True)

main_e = embeddings[0]
for i, emb in enumerate(embeddings):
    # print(emb)
    cosine_similarity = np.dot(main_e, emb) / (np.linalg.norm(main_e) * np.linalg.norm(emb))
    print(f"Similarity with {i}: {cosine_similarity}")



