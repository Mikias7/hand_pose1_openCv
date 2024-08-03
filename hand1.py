import mediapipe as mp
import cv2
import numpy as np
import time


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


cap = cv2.VideoCapture(2)

def draw_finger_angles(image,joint_list):
    
    for joint in joint_list:
        a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) 
        b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y])
        c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y])

        radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1] - b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)

        if angle > 180.0:
            angle = 360-angle

        cv2.putText(image, str(round(angle,2)), tuple(np.multiply(b,[640,480]).astype(int)),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

        
        

    return image

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip on horizontal
        image = cv2.flip(image, 1)
        
        # Set flag
        image.flags.writeable = False
        
        # Detections
        results = hands.process(image)
        
        # Set flag to true
        image.flags.writeable = True
        
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        joint_list = [[8,6,5],[12,10,9],[16,14,13]]

        

        # Detections
        print(results)
        
        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                         )
            
            draw_finger_angles(image,joint_list)
          

        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

joint_list = [[8,6,5],[12,10,9],[16,14,12]]



