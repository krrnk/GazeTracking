from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import joblib


def load_model(model_path):
    return joblib.load(model_path)


def predict(model, data):
    return model.predict( [ [data] ] )


if __name__ == '__main__':
    model = load_model('./model.joblib')
    
    import cv2
    from gaze_tracking import GazeTracking
    from eye_tracking import gaze_track

    gaze = GazeTracking()
    webcam = cv2.VideoCapture(0)

    while True:
        gaze_vel = gaze_track(gaze, webcam)
        if type(gaze_vel) is float:
            print(predict(model, gaze_vel))
        if cv2.waitKey(1) == 27:
            break

    webcam.release()
    cv2.destroyAllWindows()
