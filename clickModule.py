import cv2
import numpy as np


def mousePoints(event, x, y, flags, params):
    """Find the coordinates of the where the user has clicked"""
    if event == cv2.EVENT_LBUTTONDOWN:
        return(x,y)

