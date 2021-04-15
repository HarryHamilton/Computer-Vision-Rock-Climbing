import time
import mediapipe as mp
import cv2


class PoseDetector:
    def __init__(self, mode=False, up_body=False, smooth=True, detect_con=0.5, track_con=0.5):

        self.mode = mode
        self.up_body = up_body
        self.smooth = smooth
        self.detect_con = detect_con
        self.track_con = track_con

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.up_body, self.smooth, self.detect_con, self.track_con)

    def find_pose(self, img, draw=True):
        """Places landmarks on the users body and connects them with lines."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converts to webcam input to RGB
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                       self.mpPose.POSE_CONNECTIONS)  # Draws the lines on the body
        return img

    def find_position(self, img, draw=True):
        """Finds the location of the specified landmark. In this case, we are recording the position
        of the right hand index finger and repeatedly storing its position in an array.
        """
        if self.results.pose_landmarks:
            lm_list = []
            for id, lm in enumerate(
                    self.results.pose_landmarks.landmark):  # Prints the positions of the landmarks into console
                h, w, c = img.shape
                # Calc position of landmark
                cx, cy = int(lm.x * w), int(lm.y * h)

                lm_list.append([id, cx, cy])  # Adds the pos of the landmark to the list
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)  # Makes the circles of landmarks bigger
        return lm_list


def hit_or_miss(lm_list, targets):
    # TODO: Make this not hardcoded
    """Checks if the landmark is in the target location. This works by
    taking in the current location of the landmark (right hand) and comparing
    it to the target coordinate that has been passed in via parameters. To make
    the game easier, the users right hand is able to be in a range of + or -
    50 pixels from the target.

    targets[x][0] = x coordinate of the current set of coordinates
    targets[x][1] = y coordinate of the current set of coordinates
    """
    status = "MISS"
    for x in range(len(targets)):
        if (lm_list[20][1] > (targets[x][0] - 50)) and (lm_list[20][1] < (targets[x][0] + 50)) and \
                (lm_list[20][2] > (targets[x][1] - 50)) and (lm_list[20][2] < (targets[x][1] + 50)):
            status = "------------------------- HIT -----------------------"
    return status


def main():
    cap = cv2.VideoCapture(0)  # Takes in webcam input
    p_time = 0
    detector = PoseDetector()

    # Array that stores the targets.
    targets = [[728, 485], [325, 458]]

    while True:
        success, img = cap.read()
        img = detector.find_pose(img)
        lm_list = detector.find_position(img)

        if len(lm_list) != 0:
            print(lm_list[20])  # Print coordinates of position 20, i.e. the right index finger
            print(hit_or_miss(lm_list, targets))  # Prints whether landmark is in target area
            cv2.circle(img, (lm_list[20][1], lm_list[20][2]), 15, (0, 0, 255), cv2.FILLED)  # Adds red dot to right hand

        # Calculates FPS and displays
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, str(int(fps)), (100, 100), cv2.FONT_ITALIC, 3,
                    (255, 0, 255), 10)

        # Displays the actual window
        cv2.imshow("Body Tracker", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
