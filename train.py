import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import joblib

def load_df(file_path):
    return pd.read_csv(file_path)


def train(X, y):
    decisionTree = DecisionTreeClassifier()
    return decisionTree.fit(X, y)


def save_model(model, file_path):
    joblib.dump(model, file_path)


if __name__ == '__main__':
    df = load_df('./train_data.csv')
    X = df[['DATA']]
    y = df['CLASS']
    
    dTree = train(X, y)

    save_model(dTree, './model.joblib')
