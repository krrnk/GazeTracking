"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""
import cv2
import numpy as np
from GazeTracking.gaze_tracking import GazeTracking


prevVel = 0


def middle_point(coords_A, coords_B):
    if type(coords_A) and type(coords_B) is tuple:
        x_coord = (coords_A[0] + coords_B[0]) * 0.5
        y_coord = (coords_A[1] + coords_B[1]) * 0.5
        return int(x_coord), int(y_coord)
    else:
        return None 


def draw_middle_point(middle_point, frame):
    cv2.circle(frame, middle_point, radius=4, color=(0, 0, 255), thickness=-1)
    cv2.putText(frame, "Middle point: " + str(middle_point), (90, 195), 
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)


def velocity(value):
    global prevVel
    velocity = value - prevVel
    prevVel = value
    return velocity


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
    elif gaze.is_top():
        text = "Looking up"
    elif gaze.is_bottom():
        text = "Looking down"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 
                1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), 
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), 
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    gaze_centre = middle_point(left_pupil, right_pupil)
    draw_middle_point(gaze_centre, frame)

    avgVel = None 
    if type(gaze_centre) is tuple:
        avgVel = ( velocity(gaze_centre[0]) + velocity(gaze_centre[1]) ) * 0.5
    else:
        avgVel = None
    
    cv2.imshow("Demo", frame)

    return avgVel, gaze.is_right(), gaze.is_left(), gaze.is_top(), gaze.is_bottom(), gaze.is_center() 



if __name__ == '__main__':
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)

    while True:
        gaze_Vel = gaze_track(gaze, webcam)
        if type(gaze_Vel) is float:
            print(gaze_Vel)
        
        if cv2.waitKey(1) == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()

