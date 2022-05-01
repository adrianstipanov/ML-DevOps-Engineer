
"""
The module performs diagnostic tests related to the model, as well as the data
"""

import pickle
import logging
import pandas as pd
import subprocess
import timeit
import os
import sys
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
output_folder_path = os.path.join(config['output_folder_path'])

finaldata = pd.read_csv(os.path.join(output_folder_path, 'finaldata.csv'))


# Function to get model predictions
def model_predictions(dir_path = test_data_path, file_path = 'testdata.csv'):

    # Load model
    with open(os.path.join(prod_deployment_path, "trainedmodel.pkl"), 'rb') as model:
        model = pickle.load(model)

    # Read data
    testdata = pd.read_csv(os.path.join(dir_path, file_path))
    X_test = testdata.iloc[:, 1:]
    y_test = X_test.pop('exited').values.reshape(-1, 1).ravel()

    pred = model.predict(X_test.values)

    return pred, y_test


# Function to get summary statistics
def dataframe_summary():

    numeric = finaldata.select_dtypes(include='int64')
    stats = numeric.iloc[:, :-1].agg(['mean', 'median', 'std'])

    return stats


# Function to get missing data
def missing_data():

    nas = list(finaldata.isna().sum())
    napercents = [nas[i] / len(finaldata.index) for i in range(len(nas))]

    return napercents


# Function to get timings
def execution_time():

    timings = []
    scripts = ['ingestion.py', 'training.py']
    for process in scripts:
        starttime = timeit.default_timer()
        os.system(f'python3 {process}')
        timing = timeit.default_timer() - starttime
        timings.append(timing)

    return timings


# Function to check dependencies
def outdated_packages_list():
    outdated = subprocess.check_output(
        ['pip', 'list', '--outdated']).decode(sys.stdout.encoding)

    return str(outdated)


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    missing_data()
    execution_time()
    outdated_packages_list()
