import os
import sys
import argparse
import cv2
import numpy
import pygame
import time
import threading
import functions as f
from ultralytics import YOLO
from datetime import datetime

def play_sound():
    pygame.mixer.init()
    pygame.mixer.music.load("audio/Ding_sound.mp3")
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def main():
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
        track_results = model.track(source=[main_frame], persist=True, conf=0.5, batch=5, mode="track")  # synchronous

        cropped_stream = main_frame[int(roi[1]):int(roi[1]+roi[3]), 
                                    int(roi[0]):int(roi[0]+roi[2])]
        alert_results = model.predict(source=[cropped_stream], stream=False)
        
        cv2.imshow(f"{args.acc}", track_results[0].plot())
        cv2.imshow("Cropped Stream", alert_results[0].plot()) # Uncomment if you want to see the selected area

        global sound_bool

        if sound_bool == True:
            global last_action_time
            with action_lock:
                current_time = time.time()
                if current_time - last_action_time >= interval:
                    for results in alert_results:
                        for box in results.boxes:
                            confidence = box.conf[0]
                            class_id = box.cls[0]
                            class_name = results.names[int(class_id)]

                            if class_name == 'Person':
                                if confidence > 0.3:
                                    last_action_time = current_time
                                    try:
                                        threading.Thread(target=play_sound, daemon=True).start()
                                    except:
                                        print("Can't play sound")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            sound_bool = not sound_bool


    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    action_lock = threading.Lock()
    last_action_time = 0
    interval = 5  # seconds
    sound_bool = True
    main()

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