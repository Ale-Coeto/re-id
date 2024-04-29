import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/../'))
sys.path.append("/home/alanromero/ale/re-id/Crowd-Counting/REID_baseline_pytorch")

from REID_baseline_pytorch.reid_model import load_network, compare_images, extract_feature_from_img, get_structure
from REID_baseline_pytorch.extract_emb_torchreid import extract_features_torchreid
from ultralytics import YOLO
import torch.nn as nn
from PIL import Image
import numpy as np
import torch
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from feature_extractor import *
import collections

'''
Script to extract embeddings from people tracked in a video

It uses the swin model and 4 osnet models to extract embeddings from the people detected in the video.
The embeddings, images, and metadata are saved in a pickle file.
The script processes the first 10000 frames of the video.

Sampling method: pick detection with biggest area from each tracklet.

'''

model = YOLO('yolov8n.pt')
WIDTH_THRESHOLD = 0.1

# Object of track_id: {area, embedding} (sampled embeddings for clustering)
embeddings = {}

# Object of track_id: {image}
imgs = {}

# Object of track_id: {[customObject0 ... customObject0N]}
# customObject = {framei, embeddingi, bboxi}
tracklets = collections.defaultdict(list)

torchreid_extractors = []

model_names = ['osnet_x1_0','osnet_ibn_x1_0','osnet_ain_x1_0','osnet_x1_0']
model_paths = ["/home/alanromero/ale/re-id/Crowd-Counting/testsOscar/models/osnet_x1_0_duke_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth",
               "/home/alanromero/ale/re-id/Crowd-Counting/testsOscar/models/osnet_ibn_ms_m_c.pth.tar",
               "/home/alanromero/ale/re-id/Crowd-Counting/testsOscar/models/osnet_ain_ms_m_c.pth.tar",
               "/home/alanromero/ale/re-id/Crowd-Counting/testsOscar/models/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"
]

for model_n, model_path in zip(model_names, model_paths):
    extractor = FeatureExtractor(
        model_name= model_n,
        model_path= model_path,
        device='cuda'
    )  
    torchreid_extractors.append(extractor)

def extract_embeddings(path, model_swin):
    print("Extracting embeddings from ", path)

    video = cv2.VideoCapture(path)
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    init_size = len(embeddings)
    counter = 0
    

    for i in tqdm(range(frames), desc='Processing frames'):
            
            if counter > 10000:
                break
            counter += 1
                    
            # Get the frame from the camera
            success, frame = video.read()
            if not success:
                continue

            # Get the results from the YOLOv8 model
            results = model.track(frame, persist=True, tracker='bytetrack.yaml', classes=0, verbose=False) #could use bytetrack.yaml
            
            # Get the bounding boxes and track ids
            boxes = results[0].boxes
            track_ids = []

            try:
                track_ids = results[0].boxes.id.int().cpu().tolist()
            except Exception as e:
                track_ids = []

            # Analyze each detection (save detection with biggest area)
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()]

                area = abs(x2 - x1) * abs(y2 - y1)
                if track_id not in embeddings or area > embeddings[track_id]['area']:
                                        
                    # Crop the image 
                    cropped_image = frame[y1:y2, x1:x2]
                    
                    # Convert BGR to RGB format
                    rgb_image = cropped_image[..., ::-1]
                    
                    pil_image = Image.fromarray(rgb_image)
                        
                    imgs[track_id] = rgb_image

                    # Get embeddings
                    with torch.no_grad():
                        feature_swin = extract_feature_from_img(pil_image, model_swin).detach().numpy()[0]
                    
                    # feature_osnet = extract_features_torchreid(pil_image)
                    
                    torch_reid_features = []
                    for extractor in torchreid_extractors:
                        feature = extract_features_torchreid(pil_image, custom_extractor=extractor)
                        torch_reid_features.append(feature)

                    all_features = np.concatenate(([feature_swin], torch_reid_features), axis=0)
                    # Check that the image has the correct color format
                    # dump_img_path = f"./results/images/{track_id}_{np.random.randint(low=0, high=1000)}.jpg"
                    # pil_image.save(dump_img_path)
                    
                    feature = np.concatenate(all_features,axis=0)
                    embeddings[track_id] = {'area': area, 'embedding': feature}

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                tracklets[track_id].append({'frame': i, 'embedding': feature, 'bbox': [x1, y1, x2, y2]})

    video.release()
    cv2.destroyAllWindows()
    print(f"Extracted {len(embeddings) - init_size} embeddings from {path}")


def extract_from_path(folders, model):
    for folder in folders:
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            extract_embeddings(path, model)


def main():    
    # folders = ["../camera1", "../camera2"]
    folders = ["../S001/c001"] # Test using competition video

    structure = get_structure()
    model_swin = load_network(structure)
    model_swin.classifier.classifier = nn.Sequential()

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using cuda")
        model_swin = model_swin.cuda()

    extract_from_path(folders, model_swin)
    print(f"Final embeddings file size: {len(embeddings)}")
    
    out_dir = "./results"
    out_dir_emb = os.path.join(out_dir, "embeddings")
    out_dir_tracklets = os.path.join(out_dir, "tracklets")
    
    os.makedirs(out_dir_emb, exist_ok=True)
    os.makedirs(out_dir_tracklets, exist_ok=True)
    
    out_path = os.path.join(out_dir_emb, "SwinOsnet.pickle")
    out_path_tracklets = os.path.join(out_dir_tracklets, "tracklets.pickle")
    out_path_images = os.path.join(out_dir, "images.pickle")
    
    output = embeddings
    
    with open(out_path, "wb") as f:
        pickle.dump(output, f)
    
    with open(out_path_images, "wb") as f:
        pickle.dump(imgs, f)
    
    with open(out_path_tracklets, "wb") as f:
        pickle.dump(tracklets, f)


if __name__ == '__main__':
    print("Running feature extraction")
    main()