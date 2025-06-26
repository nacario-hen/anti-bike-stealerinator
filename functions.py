import sys
import ast
import cv2
import os
from ultralytics import YOLO

# Function for setting arguments. (Common for all)
def get_parser(parser):
    parser.add_argument('--acc', help='CCTV Account Name', required=True)
    parser.add_argument('--pw', help='CCTV Password', required=True)
    parser.add_argument('--addr', help='IP Address', required=True)
    parser.add_argument('--model', help='Model to be used [1] v1, [2] v2 ...', default=2)
    parser.add_argument('--width', help='Width to be displayed', default=640)
    parser.add_argument('--height', help='Height to be displayed', default=480)
    return parser.parse_args()

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

# Read stream
def read_stream(source):
    ret, frame = source.read()
    if not ret:
        sys.exit("Error: Could not read frame.")
    return ret, frame

# Open a video stream
def get_stream(source):
    try:
        cap = cv2.VideoCapture(source)
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

# Select an area to monitor
def select_area(source, args):
    if os.path.exists('./roi.txt'):
        file = open('./roi.txt')
        text = file.readline()
        if text:
            crop = ast.literal_eval(text)
            file.close()
            return crop
    try:
        with open("roi.txt", "w") as file:
            ret, frame = source.read()
            if ret:
                area = rescale_frame(frame, args.width, args.height)
                crop = cv2.selectROI("Select the area", area)
                cv2.destroyAllWindows()
                file.write(f"{crop}")
                file.close()
                return crop
    except Exception as e:
        print(f"What hafen, Fython? {e} right?")