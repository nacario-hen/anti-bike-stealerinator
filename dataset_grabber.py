import os
import cv2
import sys
import time
import threading
import argparse
import functions as f

action_lock = threading.Lock()
last_action_time = 0
interval = 5  # seconds

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--acc', required=True)
parser.add_argument('--pw', required=True)
parser.add_argument('--addr', required=True)
parser.add_argument('--count', default=1000)
parser.add_argument('--path', default="dataset_files")
parser.add_argument('--prefix', default="data")
args = parser.parse_args()

RTSP = "rtsp://" + args.acc + ":" + args.pw + "@" + args.addr

f.create_folder(args.path)
cap = f.get_stream(RTSP)
image_count = 0
skip = 0
local_path = './' +args.path
try:
    ret, frame = cap.read()
    frame = f.rescale_frame(frame, 640, 480)
except:
    sys.exit("Can't read stream")

while image_count < int(args.count):
    try:
        ret, frame = cap.read()
        frame = f.rescale_frame(frame, 640, 480)
    except Exception as e:
        print(f"Issue encountered:{e}. Continuing program")

    with action_lock:
        current_time = time.time()
        if current_time - last_action_time >= interval:
            cv2.imwrite(os.path.join(local_path, (str(args.prefix) + str(image_count) + ".png")), frame)
            print("Added",args.path,args.prefix,image_count, ".png")
            last_action_time = current_time
            image_count = image_count + 1
            
print("Finished gathering dataset. Closing program")
