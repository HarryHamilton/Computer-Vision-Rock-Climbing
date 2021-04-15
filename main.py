import cv2
import time
import PoseModule as pm
import clickModule as cm

# Initialise variables required throughout the program
cap = cv2.VideoCapture(0)  # Takes in webcam input
pTime = 0
detector = pm.poseDetector()


while True:
    success, img = cap.read()
    img = detector.find_pose(img)
    lm_list = detector.find_position(img)

    # The program will only display the landmark (right hand index finger) if it is detected
    if len(lm_list) != 0:
        cv2.circle(img, (lm_list[20][1], lm_list[20][2]), 15, (0, 0, 255), cv2.FILLED)  # Mark landmark w/ red dot

    # Calculates and displays FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_ITALIC, 3,
                (255, 0, 0), 3)

    # Displays the actual window
    cv2.imshow("Body Tracker", img)
    cv2.waitKey(1)


