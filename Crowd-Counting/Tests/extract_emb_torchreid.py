'''
extract ReID features from testing data.
'''
import os
import argparse
import os.path as osp
import numpy as np
# import torch
# import time
import torchvision.transforms as T
from scipy.spatial.distance import cosine
from PIL import Image
# import sys
# import torchreid
from feature_extractor import *

def make_parser():
    parser = argparse.ArgumentParser("reid")
    # parser.add_argument("root_path", type=str, default=None)
    return parser

model_p = "/home/alanromero/ale/vision/Crowd-Counting/models/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"
# model_p = "/home/alanromero/ale/vision/Crowd-Counting/models/osnet_x0_75_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"
# model_p = "/home/alanromero/ale/vision/Crowd-Counting/models/osnet_x0_5_market_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pth"
# model_p = "/home/alanromero/ale/vision/Crowd-Counting/models/hacnn_market_xent.pth.tar"
# model_p = "/home/alanromero/ale/vision/Crowd-Counting/models/mlfn_market_xent.pth.tar"
# model_p = "/home/alanromero/ale/vision/Crowd-Counting/models/mobilenetv2_1dot0_market.pth.tar"

# model_name = 'osnet_x1_0'
# model_name = 'osnet_x0_75'
# model_name = 'osnet_x0_5'
# model_name = 'hacnn'
# model_name = 'mlfn'
model_name = 'mobilenetv2_x1_0'

# model_names = ['osnet_x1_0','osnet_ibn_x1_0','osnet_ain_x1_0','osnet_x1_0','osnet_x1_0']
    

val_transforms = T.Compose([
    T.Resize([256, 128]),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

extractor = FeatureExtractor(
            model_name=model_name,
            model_path=model_p,
            device='cuda'
)  

def extract_features(cropped_image):
    img_crop = val_transforms(cropped_image.convert('RGB')).unsqueeze(0)
    feature = extractor(img_crop).cpu().detach().numpy()[0]
    return feature

def compare_images(features1, features2, threshold=0.5):

    similarity_score = 1 - cosine(features1, features2)
    
    # Compare similarity score with threshold
    if similarity_score >= threshold:
        return True  # Images are considered to be of the same person
    else:
        return False  # Images are considered to be of different persons

def get_cosine_score(features1, features2):
    
    return -1*(1 - cosine(features1, features2))
