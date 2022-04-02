"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
import numpy as np
from gaze_tracking import GazeTracking
from pythonosc.udp_client import SimpleUDPClient


def gaze_track(gaze, webcam):
    # We get a new frame from the webcam
    _, frame = webcam.read()

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Demo", frame)

    return left_pupil, right_pupil


if __name__ == '__main__':
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)
    
    ip = '127.0.0.1'
    port = 5005
    client = SimpleUDPClient(ip, port)

    while True:
        left_pupil, right_pupil = gaze_track(gaze, webcam)
        
        if type(left_pupil) and type(right_pupil) is tuple:
            left_coords = [item.tolist() for item in left_pupil]
            right_coords = [item.tolist() for item in right_pupil]
            client.send_message('/gaze_track', [left_coords, right_coords])


        if cv2.waitKey(1) == 27:
            break
    
    webcam.release()
    cv2.destroyAllWindows()
