"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
import numpy as np
import csv
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from gaze_tracking import GazeTracking
from pythonosc.udp_client import SimpleUDPClient

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


def write_csv(file_path, data, label):
    with open(file_path, 'a') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows( [ [data, label] ] )


def load_df(file_path):
    return pd.read_csv(file_path)


def train(X, y):
    decisionTree = DecisionTreeClassifier()
    return decisionTree.fit(X, y)


def predict(decisionTree, data):
    return decisionTree.predict( [ [data] ] )


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

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 
                1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), 
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), 
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    gaze_focus = middle_point(left_pupil, right_pupil)
    draw_middle_point(gaze_focus, frame)

    cv2.imshow("Demo", frame)

    return gaze_focus



if __name__ == '__main__':
    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)

    ip = '127.0.0.1'
    port = 5005
    client = SimpleUDPClient(ip, port)

    action = input('train or predict: ')

    if action == 'train':
        train_label = input("Number the lable to train: ")
    elif action == 'predict':
        df = load_df('./train_data.csv')
        X = df[['DATA']]
        y = df['CLASS']
    else:
        raise ValueError('unknown action!\nplease restart the program and select train or predict')

    dTree = train(X, y)

    while True:
        gaze_centre = gaze_track(gaze, webcam)

        if type(gaze_centre) is tuple:
            avgVel = ( velocity(gaze_centre[0]) + velocity(gaze_centre[1]) ) * 0.5
            client.send_message('/gaze_vel', avgVel)
            
            if action == 'train':
                write_csv('./train_data.csv', avgVel, train_label)
            elif action == 'predict':
                print(predict(dTree, avgVel))

        if cv2.waitKey(1) == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()
