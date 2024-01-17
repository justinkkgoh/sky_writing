import cv2
import mediapipe as mp
import easyocr
import numpy as np 
import torch

from helper_fn import  sky_drawing, relative_elb_angle, char_predict, easyocr_predict
from vision_model import EMNIST_modelV3

# Minimum distance between index and middle fingers to initiate drawing
# Change this value depending on HOW FAR the user is from the camera 
THRESHOLD = 0.045

# Angle for arm to be considered bent
ANGLE_THRESHOLD = 160 

# Initializing variables for mediapipe 
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize webcam capture
cap = cv2.VideoCapture(0)

draw_mode = False 
elbow_det = False 


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    key = cv2.waitKey(1) & 0xFF
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    
    # Extracting hand landmarks 
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    if key == ord('d'):
        draw_mode = True 
        finger_loc = []
        
    if draw_mode:
        
        canvas = sky_drawing(finger_loc, 
                    THRESHOLD, 
                    index_finger_tip, 
                    middle_finger_tip, 
                    frame,
                    )
        
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        

        

    # Extracting pose landmarks 
    pose_results = pose.process(rgb_frame)

    if key == ord('e'):
        elbow_det = True

    if pose_results.pose_landmarks and elbow_det == True: 

        relative_angle = relative_elb_angle(pose_results, mp_pose)
        print(f"Detected Angle: {relative_angle}")


        if relative_angle < ANGLE_THRESHOLD: 
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(106,15,172), thickness=4, circle_radius = 5),
                                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=4, circle_radius = 10)
                                    )
            
        else: 
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(106,15,172), thickness=4, circle_radius = 5),
                                mp_drawing.DrawingSpec(color=(255,100,10), thickness=4, circle_radius = 10)
                                )


    # 'Clear', to exit drawing mode
    if key == ord('c'):
        draw_mode = False 
        predict_mode = False 
        elbow_det = False 


    # Function to predict characters, via Your own trained model 
    if key == ord("r"):

        if 'canvas' not in locals():
            print("Must draw something before predicting")

        else: 
            char_predict(canvas)

    # Function to predict characters, via easyOCR 
    if key  == ord("w"): 
        if 'canvas' not in locals():
            print("Must draw something before predicting")
        else: 
           detected_im = easyocr_predict(canvas)

    cv2.imshow('Hand Tracking', frame)

    # Quit program
    if cv2.waitKey(1)  == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
