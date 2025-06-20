import sys
import cv2
import os
from ultralytics import YOLO

# Function for scaling the video
def rescale_frame(frame, width, height):    # works for image, video, live video
    dimension = (width, height)
    return cv2.resize(frame, dimension, interpolation=cv2.INTER_AREA) #cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

# Model for Object Detection
def get_model(model_path):
    try:
        model = YOLO(model_path)
        print("Model detected.")
        return model
    except:
        sys.exit("Model not detected. Terminating program")

# Open a video stream
def get_stream(footage):
    try:
        cap = cv2.VideoCapture(footage)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("Stream Detected.")
        return cap
    except:
        sys.exit("Stream not detected. Terminating program")

# Create a folder
def create_folder(folder_name):
    try:
        os.mkdir(folder_name)
        print("Directory '", folder_name, "' created successfully")
    except FileExistsError:
        print("Directory '", folder_name, "' already existing")
    except Exception as e:
        print("Encountered an error: ", e)
