import cv2
import time
import PoseModule as pm
import clickModule as cm

cap = cv2.VideoCapture(0)  # Takes in webcam input
pTime = 0
detector = pm.poseDetector()

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        print(lmList[20])  # Displays position 20, i.e. the right index finger, in console
        cv2.circle(img, (lmList[20][1], lmList[20][2]), 15, (0, 0, 255), cv2.FILLED)  # Places dot on the landmark we are tracking
    # Calculates FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_ITALIC, 3,
                (255, 0, 0), 3)

    # Displays the actual window
    cv2.imshow("Body Tracker", img)
    cv2.waitKey(1)


