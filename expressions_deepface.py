import warnings
warnings.filterwarnings("ignore")
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from deepface import DeepFace
import glob 
import cv2
import numpy as np
import tqdm 

files = glob.glob("BagOfLies/Finalised/User_*/run_*/video.mp4")

for file in tqdm.tqdm(files):
    print(f"Parsing {file}...")
    cap = cv2.VideoCapture(file)
    frames = []
    expressions = []
    # Loop through each frame of the video
    for i in tqdm.tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        # Read the next frame
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to BGR
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Append the frame to the list
        
        objs = DeepFace.analyze(img_path = frame_bgr, 
            actions = ['emotion'], enforce_detection=False)
    
        expressions.append(objs[0]['dominant_emotion'])

    # Convert the list of frames to a numpy array

    with open(file.replace("video.mp4", "expressions.txt"), "w") as f:
        for expression in expressions:
            f.write(expression + "\n")
    # Release the video file
    cap.release()
