import cv2
from ultralytics import YOLO
import os


'''
Script to extract people detection images from a video
'''

model = YOLO('yolov8n.pt')
WIDTH_THRESHOLD = 0.1

def get_detections(path, name, id):
    cap = cv2.VideoCapture(path)
    counter = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    desired_percentage = 0.5  # Desired percentage of frames to process"
    frames_to_process = int(total_frames * desired_percentage)
    step_size = max(total_frames // frames_to_process, 1)
    desired_frames = 100  # Number of frames you want to process
    step_size = max(total_frames // desired_frames, 1)

    output_folder = f"{name}_cropped"
    os.makedirs(output_folder, exist_ok=True)

    print("Processing ", path)
    print("Frames: ", frames_to_process)
    images = 0

    while cap.isOpened():
        # print(counter)
        success, frame = cap.read()
        counter += 1

        if success:

            image_width = frame.shape[1]

            if counter % step_size == 0:
                results = model.track(source=frame, persist=True, classes=0, show=False, verbose=False)
                bboxes = results[0].boxes.xywh.cpu().tolist()
                # print(len(bboxes))
                for bbox in bboxes:

                    x = int(bbox[0])
                    y = int(bbox[1])
                    w = int(bbox[2])
                    h = int(bbox[3]) 

                    if w >= WIDTH_THRESHOLD * image_width:
                        x1 = int(x - w / 2)
                        y1 = int(y - h / 2)
                        x2 = int(x + w / 2)
                        y2 = int(y + h / 2)
                        cropped_image = frame[y1:y2, x1:x2]
                        # print("new img")
                        frame_path = os.path.join(output_folder, f'vid{id}_frame_{counter}.jpg')
                        cv2.imwrite(frame_path, cropped_image)
                        # images += 1

        else:
        # End of video
            break
        

    cap.release()
    cv2.destroyAllWindows()


def get_detections_from_folder(folder, name):
        
    for i, filename in enumerate(os.listdir(folder)):
        path = os.path.join(folder, filename)
        get_detections(path, name, i)
        # print(filename, " : " , path)


folder1 = "./camera1"
folder2 = "./camera2"

get_detections_from_folder(folder1,"cam1")
get_detections_from_folder(folder2, "cam2")
# get_detections("cp_2.mp4", "test", 1)

