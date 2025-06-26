import os
import sys
import argparse
import cv2
import numpy
import time
import functions as f
from ultralytics import YOLO
from datetime import datetime
from playsound import playsound

args = f.get_parser(argparse.ArgumentParser())
RTSP = "rtsp://" + args.acc + ":" + args.pw + "@" + args.addr
MODEL_PATH = f'./my_model_v{args.model}/my_model.pt' # temporary. Remove args model if final true model is achieved/acquired

cap = f.get_stream(RTSP)
model = f.get_model(MODEL_PATH)
roi = f.select_area(cap, args)

# Display stream until break (q key) is pressed
while True:

    ret, main_frame = f.read_stream(cap)
    main_frame = f.rescale_frame(main_frame, args.width, args.height)
    # Run YOLO inference in tracking mode. See kwargs for additional args
    results = model.track(source=[main_frame], persist=True, conf=0.3, batch=1, mode="track")  # synchronous

    ret, alert_frame = f.read_stream(cap)

    cropped_stream = f.rescale_frame(alert_frame, args.width, args.height)
    cropped_stream = cropped_stream[int(roi[1]):int(roi[1]+roi[3]), 
                                    int(roi[0]):int(roi[0]+roi[2])]
    alert_results = model.predict(source=[cropped_stream], stream=False)

    cv2.imshow(f"{args.acc}", results[0].plot())
    cv2.imshow("Cropped Stream", alert_results[0].plot())

    for results in alert_results:
        for box in results.boxes:
            confidence = box.conf[0]
            class_id = box.cls[0]
            class_name = results.names[int(class_id)]

            if class_name == 'Person':
                if confidence > 0.5:
                    try:
                        playsound('./audio/Ding_sound.mp3')
                    except:
                        print("Can't play sound")

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