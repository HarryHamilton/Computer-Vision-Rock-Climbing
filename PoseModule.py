import time
import mediapipe as mp
import cv2


class poseDetector():
    def __init__(self, mode = False, upBody = False, smooth = True, detectCon = 0.5, trackCon = 0.5):

        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectCon = detectCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectCon, self.trackCon)

    def findPose(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converts to rgb
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                       self.mpPose.POSE_CONNECTIONS)  # Draws the lines on the body
        return img

    def findPosition(self, img, draw=True):
        if self.results.pose_landmarks:
            lmList = []
            for id, lm in enumerate(self.results.pose_landmarks.landmark):  # Prints the positions of the landmarks into console
                h, w, c = img.shape
                #print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)

                lmList.append([id, cx, cy])  # Adds the pos of the landmark to the list
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)  # Makes the circles on landmarks bigger
        return lmList




def main():
    cap = cv2.VideoCapture(0)  # Takes in webcam input
    pTime = 0
    detector = poseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[20])  # Displays position 20, i.e. the right index finger
        # Calculates FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_ITALIC, 3,
                    (255, 0, 0), 3)
        # Displays the actual window
        cv2.imshow("Body Tracker", img)
        cv2.waitKey(1)



if __name__ == "__main__":
    main()
