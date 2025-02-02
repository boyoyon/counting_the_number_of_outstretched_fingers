import cv2, os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

dm = mp.solutions.drawing_utils

NR_KEYPOINTS = 21
TH_ANGLE = 35
ESC_key = 27

font = cv2.FONT_HERSHEY_PLAIN
font_size = 2
font_color = (0, 255, 0)
font_pos = (10, 30)

def extract_points(hand_landmarks):

    points = []
    
    for keypoint in range(NR_KEYPOINTS):

        x = hand_landmarks.landmark[keypoint].x
        y = hand_landmarks.landmark[keypoint].y
        z = hand_landmarks.landmark[keypoint].z

        points.append(np.array((x, y, z)))

    return points

def getAngle(v0, v1):

    inner = np.inner(v0, v1)

    normV0 = np.linalg.norm(v0)
    normV1 = np.linalg.norm(v1)
    
    theta = np.arccos(inner/(normV0 * normV1 + 1e-6))

    return np.rad2deg(theta)

def main():

    hands = mp_hands.Hands(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)

    nr_fingers = 0
    prev_nr_fingers = -1

    print('Hit ESC-key to terminate')

    while cap.isOpened():

        success, image = cap.read()
        if not success:
            print('skip failed capture frame')
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.

        image.flags.writeable = False
        results = hands.process(image)
    
        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                dm.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if len(hand_landmarks.landmark) == 21:
               
                    points = extract_points(hand_landmarks)

                    try:
                        v0 = points[2] - points[0]
                    except TypeError:
                        #print('TypeError occurs at v0 = p2 - p0')
                        continue

                    try:
                        v1 = points[4] - points[2]
                    except TypeError:
                        #print('TypeError occurs at v1 = p4 - p2')
                        continue
                    
                    try:
                        v2 = points[5] - points[0]
                    except TypeError:
                        #print('TypeError occurs at v2 = p5 - p0')
                        continue
                    
                    try:
                        v3 = points[8] - points[5]
                    except TypeError:
                        #print('TypeError occurs at v3 = p8 - p5')
                        continue
                    
                    try:
                        v4 = points[9] - points[0]
                    except TypeError:
                        #print('TypeError occurs at v4 = p9 - p0')
                        continue
                    
                    try:
                        v5 = points[12] - points[9]
                    except TypeError:
                        #print('TypeError occurs at v5 = p12 - p9')
                        continue
                    
                    try:
                        v6 = points[13] - points[0]
                    except TypeError:
                        #print('TypeError occurs at v6 = p13 - p0')
                        continue
                    
                    try:
                        v7 = points[16] - points[13]
                    except TypeError:
                        #print('TypeError occurs at v7 = p16 - p13')
                        continue
                    
                    try:
                        v8 = points[17] - points[0]
                    except TypeError:
                        #print('TypeError occurs at v8 = p17 - p0')
                        continue
                    
                    try:
                        v9 = points[20] - points[17]
                    except TypeError:
                        #print('TypeError occurs at v9 = p20 - p17')
                        continue

                    nr_fingers = 0
                    if getAngle(v0, v1) < TH_ANGLE:
                        nr_fingers += 1

                    if getAngle(v2, v3) < TH_ANGLE:
                        nr_fingers += 1

                    if getAngle(v4, v5) < TH_ANGLE:
                        nr_fingers += 1

                    if getAngle(v6, v7) < TH_ANGLE:
                        nr_fingers += 1

                    if getAngle(v8, v9) < TH_ANGLE:
                        nr_fingers += 1

                    if nr_fingers != prev_nr_fingers:
                        print(nr_fingers)
                        prev_nr_fingers = nr_fingers
                        
        cv2.putText(image, '%d' % nr_fingers, font_pos, font, font_size, font_color, 2)
        cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(5) & 0xFF == ESC_key:
            break

    hands.close()
    cap.release()

if __name__ == '__main__':
    main()
