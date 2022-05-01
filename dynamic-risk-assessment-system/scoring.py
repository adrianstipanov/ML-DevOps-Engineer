"""
This module computes the scoring of the trained model
"""

import logging
import pandas as pd
import pickle
import os
from sklearn import metrics
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
artifacts_path = os.path.join(config['output_model_path'])


# Function for model scoring
def score_model():

    with open(os.path.join(artifacts_path, 'trainedmodel.pkl'), "rb") as file:
        model = pickle.load(file)

    testdata = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    X_test = testdata.iloc[:, 1:]
    y_test = X_test.pop('exited').values.reshape(-1, 1).ravel()

    pred = model.predict(X_test.values)

    f1_score = metrics.f1_score(pred, y_test)

    # Save metrics
    with open(os.path.join(artifacts_path, "latestscore.txt"), 'w') as file:
        file.write(str(f1_score))

    logging.info(f"Scoring: F1={f1_score:.2f}")

    return f1_score


if __name__ == '__main__':
    score_model()
