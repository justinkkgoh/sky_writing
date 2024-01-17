import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import torch
import easyocr
import math 
from vision_model import EMNIST_modelV3


def sky_drawing(finger_loc, THRESHOLD, index_finger_tip, middle_finger_tip, frame):
        
        """ 
        Creates an empty canvas and draws the character on that canvas based on where the user's index and middle finger is located. 

        Args:
            canvas (class 'numpy.ndarray): input image to be preprocessed 

        Returns: 
            canvas
        """
    
        canvas = np.zeros_like(frame)

        # Check if index and middle fingers are close together
        if abs(index_finger_tip.x - middle_finger_tip.x) < THRESHOLD and \
                abs(index_finger_tip.y - middle_finger_tip.y) < THRESHOLD:
            
            # Draw a circle when fingers are close together
            center = (int((index_finger_tip.x + middle_finger_tip.x) * frame.shape[1] / 2),
                    int((index_finger_tip.y + middle_finger_tip.y) * frame.shape[0] / 2))
 
            finger_loc.append(center)
          

        radius = 15  
        color = (139, 140, 0)  # dark blue 
        thickness = -1 
            
        for i in range(len(finger_loc)-1):
            cv2.circle(frame, finger_loc[i], radius, color, thickness)
            cv2.circle(canvas, finger_loc[i], radius, color, thickness)

        return canvas    


def relative_elb_angle(pose_results, mp_pose):
        
        """ 
        Calculates the angle of shoulder to elbow joint and elbow to wrist joint with respect to the x-axis.

        Args:
            relative_angle (class 'float'): Angle of elbow

        Returns: 
            relative_angle
        """
        
        arm_landmark = pose_results.pose_landmarks.landmark

        # Extracting landmarks for shoulder, elbow and wrist 
        shoulder_landmark = arm_landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        elbow_landmark = arm_landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
        wrist_landmark = arm_landmark[mp_pose.PoseLandmark.LEFT_WRIST]

        # Calculating angles of shoulder to elbow joint and elbow to wrist joint, with respect to the x-axis 
        shoulder_angle = math.degrees(math.atan2(elbow_landmark.y - shoulder_landmark.y, elbow_landmark.x - shoulder_landmark.x))
        elbow_angle = math.degrees(math.atan2(wrist_landmark.y - elbow_landmark.y, wrist_landmark.x - elbow_landmark.x))
        
        relative_angle = (elbow_angle - shoulder_angle) + 180 

        # Printing detected angles 
        print(f"Detected Angle: {relative_angle}")

        return relative_angle 


def EMNIST_preprocess(frame): 
    """ 
    Preprocesses an input image in the way described in the paper "EMNIST: an extension of MNIST handwritten letters"

    Args:
        frame (class 'numpy.ndarray): input image to be preprocessed 

    Returns: 
        resize (class 'numpy.ndarray): preprocessed image
    """

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    

    print("Gray shape:", gray_image.shape)

    _, binary_image = cv2.threshold(gray_image, 50,255, cv2.THRESH_BINARY)
    gaus = cv2.GaussianBlur(binary_image, (5,5), sigmaX=1, sigmaY=1)
    #print("Gaussian Type: ", type(gaus))

    # Finding image contours 
    contours, _ = cv2.findContours(gaus, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(largest_contour)
    roi = gray_image[y:y+h, x:x+w]

    # Centering the image 
    side = max(w,h)
    canvas = np.zeros((side,side), dtype=np.uint8)


    start_x = (side - w) // 2 
    start_y = (side-h) // 2 
    canvas[start_y:start_y+h, start_x:start_x+w] = roi 

    padding = cv2.copyMakeBorder(canvas, 20,20,20,20, cv2.BORDER_CONSTANT,value=[0,0,0])

    # Resize 
    resize = cv2.resize(padding, (28,28), interpolation=cv2.INTER_CUBIC)

    return resize 


def char_predict(canvas):
    """ 
    Performs preprocesssing on input image
    Something MUST be drawn on canvas prior to calling this 
    Image is then sent to trained model for inference 

    Args:
        canvas (class 'numpy.ndarray): input image to be sent to model

    Returns: 
        prints predicted character 
    """

    # Preprocessing the image 
    prepro_im = EMNIST_preprocess(canvas)
    print('Image processed')

    # Adding batch and channel dim to preprocessed image 
    image_np = np.expand_dims(np.expand_dims(prepro_im, axis=0), axis=0)
    image_tensor = torch.tensor(image_np, dtype=torch.float32)

    if torch.cuda.is_available(): 
        DEVICE = 'gpu'
        print('GPU selected')
    else: 
        DEVICE = 'cpu'
        print('CPU selected')

    # Load weights and Instantiate model
    weight_path = './saved_weights/Exp_number2.pt'
    model = EMNIST_modelV3(input_shape=1, 
                    hidden_units=16, 
                    output_shape=47).to(DEVICE)
    
    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.eval()
    
    print('Model Loaded')

    # Perform inference 
    with torch.no_grad():
        output = model(image_tensor)
        _, preds_tensor = torch.max(output, 1)
        preds = np.squeeze(preds_tensor.numpy())

    print(f"Predicted character: ",chr(preds+65))


def easyocr_predict(canvas):

    """ 
    Performs optical character recognition based on easyOCR framework

    Args:
        canvas (class 'numpy.ndarray): input image to be sent to model

    Returns: 
        prints predicted character or text
    """

    print("Detecting text...")
    gpu_flag = True if torch.cuda.is_available() else False
    print('CUDA present:', gpu_flag)

    reader = easyocr.Reader(['en'], gpu=gpu_flag)
    predictions = reader.readtext(canvas)
    font = cv2.FONT_HERSHEY_SIMPLEX 
        
    for detection in predictions:
        top_left = tuple(int(val) for val in detection[0][0])
        bot_right = tuple(int(val) for val in detection[0][2])

        text = detection[1]
        print(f"Detected text: {text}")        
        image = cv2.rectangle(canvas,top_left, bot_right,[0,255,0],2)
        image = cv2.putText(canvas, text, top_left, font, 2,[255,255,0],2,cv2.LINE_AA)
    
    if 'image' in globals(): 
        # If text was not detected, new window will not show up OR
        # window will displace most recent detected text 
        cv2.imshow('Detected Text', image)




