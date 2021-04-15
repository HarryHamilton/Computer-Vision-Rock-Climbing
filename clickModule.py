import cv2


def mouse_points(event, x, y, flags, params):
    """Find the coordinates of the where the user has clicked"""
    if event == cv2.EVENT_LBUTTONDOWN:
        return (x, y)
