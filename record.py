import csv

def write_csv(file_path, data, label):
    with open(file_path, 'a') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerows( [ [data, label] ] )


if __name__ == '__main__':
    train_label = input("Number the data label: ")
    
    import cv2
    from gaze_tracking import GazeTracking
    from eye_tracking import gaze_track

    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)

    while True:
        gaze_vel = gaze_track(gaze, webcam)
        if type(gaze_vel) is float:
            print(gaze_vel)
            write_csv('./train_data.csv', gaze_vel, train_label)
      
        if cv2.waitKey(1) == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()
