import cv2
import sys
import time
import argparse
from main import rescale_frame

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--acc', required=True)
parser.add_argument('--pw', required=True)
parser.add_argument('--addr', required=True)
parser.add_argument('--count', default=1000)
args = parser.parse_args()

RTSP = "rtsp://" + args.acc + ":" + args.pw + "@" + args.addr

try:
    cap = cv2.VideoCapture(RTSP)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("Stream Detected")
except:
    sys.exit("Stream not detected. Terminating program")

try:
    ret, frame = cap.read()
    image_count = 0
except:
    sys.exit("Can't read stream")

while image_count < (args.count + 1):
    cv2.imwrite("data%d.png", image_count, frame)
    time.sleep(10)
    image_count = image_count + 1

print("Finished gathering dataset. Closing program")
