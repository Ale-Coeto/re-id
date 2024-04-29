from REID_baseline_pytorch.reid_model import load_network, compare_images, extract_feature_from_img, get_structure
from REID_baseline_pytorch.extract_emb_torchreid import extract_features_torchreid
from ultralytics import YOLO
import torch.nn as nn
from PIL import Image
import numpy as np
import torch
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

'''
Script to extract embeddings from people tracked in a video
Args:
    filevideo[str]: path to the video to track
    model: model of the detector that the tracker will use
    time: timestamp from the camera entity record
Returns: 
    [dict]: a dictionary with person entities detected from the video
'''

model = YOLO('yolov8n.pt')
WIDTH_THRESHOLD = 0.07
HEIGHT_THRESHOLD = 0.15
CONFIDENCE_THRESHOLD = 0.8
SHOW_RESULTS = False


final_imgs = []
final_swin = {}
final_osnet = {}
embeddings = []


def extract_embeddings(path, model_swin, name):
    print("Extracting embeddings from ", path)

    video = cv2.VideoCapture(path)
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # frames = 100
    prev_ids = []
    new_index = frames

    embeddings_swin = {}
    embeddings_osnet = {}
    imgs = {}

    # Process each frame
    for i in tqdm(range(frames), desc='Processing frames'):
            
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

            # Analyze each detection
            curr_ids = []
            for box, track_id in zip(boxes, track_ids):

                # Get bounding box and confidence of detection
                x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].tolist()]
                confidence = round(box.conf[0].item(), 2)
                width = abs(x2 - x1) / frame.shape[1] # Normalized width
                height = abs(y2 - y1) / frame.shape[0] # Normalized height
                
                # Extract embeddings if the detection is valid (confidence, width and height thresholds)
                if confidence > CONFIDENCE_THRESHOLD and width > WIDTH_THRESHOLD and height > HEIGHT_THRESHOLD:

                    # Crop the image 
                    cropped_image = frame[y1:y2, x1:x2]
                    rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_image)

                    # Get embeddings
                    with torch.no_grad():
                        feature_swin = extract_feature_from_img(pil_image, model_swin).detach().numpy()[0]
                    feature_osnet = extract_features_torchreid(pil_image)

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # If the track id is not in the embeddings dictionary, add it
                    if track_id not in embeddings_swin:
                        imgs[track_id] = cropped_image
                        embeddings_swin[track_id] = [feature_swin]
                        embeddings_osnet[track_id] = [feature_osnet]
                        # embeddings_swin[track_id] += feature_swin
                        # embeddings_osnet[track_id] += feature_osnet
                        
                    # Check for id switches (if a track_id is already in the dictionary but not in the previous frame)
                    elif track_id not in prev_ids:
                        new_index += 1
                        embeddings_swin[new_index] = [feature_swin]
                        embeddings_osnet[new_index] = [feature_osnet]
                        imgs[new_index] = cropped_image
                    
                    # If the track id is already in the embeddings dictionary, append the new embeddings (to later fuse the embeddings)
                    else:
                        embeddings_swin[track_id].append(feature_swin)
                        embeddings_osnet[track_id].append(feature_osnet)

                # Draw bounding box if the detection is not valid
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Update current ids
                curr_ids.append(track_id)

            prev_ids = curr_ids

            # Show results
            if SHOW_RESULTS:
                cv2.imshow("frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    
    # Fuse embeddings and concatenate different models
    for emb_swin, emb_osnet, img in zip(embeddings_swin.values(), embeddings_osnet.values(), imgs.values()):
        if len(emb_swin) <= 1:
            feature = np.concatenate((emb_swin[0], emb_osnet[0]), axis=0)
        else:
            fused_swin = np.mean(emb_swin, axis=0)
            fused_osnet = np.mean(emb_osnet, axis=0)
            feature = np.concatenate((fused_swin, fused_osnet), axis=0)

        embeddings.append(feature)
        final_imgs.append(img)

    # Release video and close windows
    video.release()
    cv2.destroyAllWindows()
    print(f"Extracted {len(embeddings_osnet)} embeddings from {name}")


# Extract embeddings from a folder
def extract_from_path(folder, model, name):
        
    for i, filename in enumerate(os.listdir(folder)):
        path = os.path.join(folder, filename)
        extract_embeddings(path, model, name)

# Main
def main():    
    folder1 = "./camera1"
    folder2 = "./camera2"
    folder3 = "./S001/c001"

    structure = get_structure()
    model_swin = load_network(structure)
    model_swin.classifier.classifier = nn.Sequential()

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using cuda")
        model_swin = model_swin.cuda()

    out_path = os.path.join('./embeddings',"Swin.npy")
   
    # extract_from_path(folder1, model_swin, "cam1")
    # extract_from_path(folder2, model_swin, "cam2")
    extract_from_path(folder3, model_swin, "testVid")


    print(f"Final embeddings file size: {len(embeddings)}")
    output = np.array(embeddings)

    print("Saving embeddings to ", out_path)
    print("size:", output.shape)
    print(output[0]) if len(output) > 0 else None
    np.save(out_path, output)

    with open("./results/images.pickle", "wb") as f:
        pickle.dump(final_imgs, f)


if __name__ == '__main__':
    print("Running feature extraction")
    main()