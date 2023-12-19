import cv2
from pyardrone import ARDrone
import mediapipe as mp
import tensorflow as tf
import network3 as trainedNetwork
import time
import numpy as np
import keyboard
import os
import keras

# # # Define variable for later use
# Saving captured data from drone and name of dataset category
isSaving = False
category_folder = ''
# Prediction outputs
predictOutputs = {
    0: 'Down',
    7: 'Up',
    1: 'Idle BG',
    2: 'Idle Person',
    3: 'Left',
    4: 'Right',
    5: 'Too Close',
    6: 'Too Far'
}
# Time calc for FPS
pTime = 0
timestamp = 0
droneControl = False

# Initiate Drone connection for video and control
drone = ARDrone()
drone.navdata_ready.wait()
print('navdata_ready')
drone.video_ready.wait()
print('video_ready')

topDown = 0
leftRight = 0
forwBack = 0

# Define drone movement based on prediction output
droneMovement = {
    0: drone.move(down=0.1),
    7: drone.move(up=0.1),
    1: 'Idle BG',
    2: 'Idle Person',
    3: drone.move(left=0.1),
    4: drone.move(right=0.1),
    5: drone.move(backward=0.1),
    6: drone.move(forward=0.1)
}

# # # Load trained model
model = trainedNetwork.load_model()
keras.utils.plot_model(model, show_shapes=True)
# Initiate Mediapipe pose and hand model for extraction
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Using openCV to read data from computer camera (for testing)
# cap = cv2.VideoCapture(0)

while True:
    result = None
    # Read data from camera (for testing)
    # success, img = cap.read()
    # Read data from DRONE camera
    img = drone.frame

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Process the frame for pose recognition
    pose_result = pose.process(imgRGB)
    # Process the frame for hand recognition
    hand_results = hands.process(imgRGB)

    # Draw pose onto image
    if pose_result.pose_landmarks:
        mpDraw.draw_landmarks(img, pose_result.pose_landmarks, mpPose.POSE_CONNECTIONS,
                              connection_drawing_spec=mpDraw.DrawingSpec(
                                  thickness=10,
                                  color=(128, 255, 128),
                                  circle_radius=5),
                              landmark_drawing_spec=mpDraw.DrawingSpec(
                                  thickness=3,
                                  color=(255, 56, 56),
                                  circle_radius=10)
                              )
    # Draw hands onto image
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS,
                                  connection_drawing_spec=mpDraw.DrawingSpec(
                                      thickness=5,
                                      color=(0, 0, 255),
                                      circle_radius=5
                                  ),
                                  landmark_drawing_spec=mpDraw.DrawingSpec(
                                      thickness=3,
                                      color=(255, 239, 128),
                                      circle_radius=5)
                                  )

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        drone.land()
        break

    # # # FPS calculation
    # Get time
    cTime = time.time()
    # Time that has passed
    fps = 1 / (cTime - pTime)
    pTime = cTime
    timestamp += 1

    # If saving is true
    if isSaving:
        # Get path for data saving
        folder_path = os.path.join(os.getcwd(), f'data/{category_folder}')
        # Create the "data" folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # Create filename by timestamp *
        filename = os.path.join(folder_path, f"{str(timestamp)}.jpg")
        # Write file
        cv2.imwrite(filename, img)

    # If saving is false
    if not isSaving:
        # resize image for prediction
        img_predict = tf.image.resize(img, [224, 224])
        img_predict = tf.reshape(img_predict, [1, 224, 224, 3])
        # predict list of outputs
        classes = model.predict(img_predict)
        # Get prediction with the most accuracy
        output = np.argmax(classes)

        result = predictOutputs.get(output)

        if keyboard.is_pressed(17):  # T - Drone has control
            droneControl = not droneControl

        print(droneControl)

        if droneControl:
            time.sleep(0.1)
            if output == 0 and topDown > -1:
                result = 'Down'
                drone.move(down=0.2)
                topDown -= 0.1

            elif output == 7 and topDown < 1:
                result = 'Up'
                drone.move(up=0.2)
                topDown += 0.1

            elif output == 1:
                result = 'Idle BG'

            elif output == 2:
                result = 'Idle Person'

            elif output == 3 and leftRight > -1:
                result = 'Left'
                # drone.move(right=0.1)
                leftRight -= 0.1

            elif output == 4 and leftRight < 1:
                result = 'Right'
                # drone.move(left=0.1)
                leftRight += 0.1

            elif output == 5 and forwBack > -1:
                result = 'Too Close'
                drone.move(backward=0.2)
                forwBack -= 0.1

            elif output == 6 and forwBack < 1:
                result = 'Too Far'
                drone.move(forward=0.1)
                forwBack += 0.1

    print(result)
    if keyboard.is_pressed(45) and not isSaving:  # N - Create new dataset based on given name
        category_folder = input("Enter category for dataset: ")
        isSaving = True

    if keyboard.is_pressed(46) and isSaving:  # M - Stop creating dataset
        isSaving = False

    if keyboard.is_pressed(38):  # J - Take off
        print('Taking Off')
        drone.takeoff()
    if keyboard.is_pressed(13):  # W - Forward
        drone.move(forward=0.2)
    if keyboard.is_pressed(1):  # S - Backward
        drone.move(backward=0.2)
    if keyboard.is_pressed(6):  # Z - Left
        drone.move(left=0.2)
    if keyboard.is_pressed(2):  # D - Right
        drone.move(right=0.2)
    if keyboard.is_pressed(12):  # Q - Up
        drone.move(up=0.2)
    if keyboard.is_pressed(14):  # E - Down
        drone.move(down=0.2)
    if keyboard.is_pressed(15):  # R - Rotation clockwise
        drone.move(cw=0.5)
    if keyboard.is_pressed(16):  # Y - Rotation counter-clockwise
        drone.move(ccw=0.5)
    if keyboard.is_pressed(40):  # K - Landing
        print('Landing')
        drone.land()

    cv2.putText(img, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
