import os
import sys
import argparse
import cv2
import numpy
import time
import functions as f
from ultralytics import YOLO
from datetime import datetime

# ARGS
parser = argparse.ArgumentParser() # Declare parser
parser.add_argument('--acc', help='CCTV Account Name', required=True)
parser.add_argument('--pw', help='CCTV Password', required=True)
parser.add_argument('--addr', help='IP Address', required=True)
parser.add_argument('--width', help='Width to be displayed', default=640)
parser.add_argument('--height', help='Height to be displayed', default=480)
args = parser.parse_args() # Parse arguments received in command line

# CONSTANTS
RTSP = "rtsp://" + args.acc + ":" + args.pw + "@" + args.addr

# Model
MODEL_PATH = './my_model/my_model.pt' # temporary

model = f.get_model(MODEL_PATH)
cap = f.get_stream(RTSP, args.width, args.height)

# Display stream until break (q key) is pressed
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Run YOLO inference in tracking mode. See kwargs for additional args
    results = model.track(source=[frame], persist=True, conf=0.3, batch=5, mode="track")  # synchronous

    annotated_frame = f.rescale_frame(results[0].plot())

    cv2.imshow("RTSP Stream", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# might be needed in the future

# w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# fps = cap.get(cv2.CAP_PROP_FPS) 
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# while True:
#     dt  = datetime.now().isoformat().replace(':','-').replace('.','-')
#     out = cv2.VideoWriter('log/output' + dt + '.mp4', fourcc, fps, (int(w),int(h)))

#     # start timer
#     start_time = time.time()

#     # Capture video from camera per 60 seconds
#     while (int(time.time() - start_time) < 60):
#         ret, frame = cap.read()
#         if ret==True:

#             #frame = cv2.flip(frame,0) # Do you want to FLIP the images?

#             out.write(frame)

#             cv2.imshow('frame',frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         else:
#             break

#     # Release everything if job is finished
#     out.release()

#     list_of_files = os.listdir('log')
#     full_path = ["log/{0}".format(x) for x in list_of_files]

#     if len(list_of_files) == 15:
#         oldest_file = min(full_path, key=os.path.getctime)
#         os.remove(oldest_file)

# cap.release()

# cv2.destroyAllWindows()


# if SAVE_FRAMES:
#     frame_filename = os.path.join(OUTPUT_FRAMES_DIR, f"frame_{frame_count:05d}.jpg")
#     cv2.imwrite(frame_filename, frame)