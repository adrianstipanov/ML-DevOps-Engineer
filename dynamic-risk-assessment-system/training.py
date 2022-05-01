"""
The module trains a Logistic Regression model
"""
import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])


# Function for training the model
def train_model():

    # use this logistic regression for training
    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class='auto',
        n_jobs=None,
        penalty='l2',
        random_state=0,
        solver='liblinear',
        tol=0.0001,
        verbose=0,
        warm_start=False)

    # Split the data into features and label
    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    X = df.iloc[:, 1:]
    y = X.pop('exited').values.reshape(-1, 1).ravel()

    model.fit(X.values, y)

    # write the trained model to your workspace in a file called
    # trainedmodel.pkl

    with open(os.path.join(model_path, 'trainedmodel.pkl'), 'wb') as file:
        pickle.dump(model, file)



if __name__ == '__main__':
    train_model()
    logging.info("Model successfully trained and saved")

