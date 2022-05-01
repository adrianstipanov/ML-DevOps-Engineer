"""
This module automates the ML model scoring and monitoring process
"""
from training import train_model
import deployment
from diagnostics import model_predictions
from sklearn.metrics import f1_score
import ingestion
import json
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

input_path = os.path.join(config['input_folder_path'])
prod_directory = os.path.join(config['prod_deployment_path'])
output_folder_path = config['output_folder_path']
artifacts_path = os.path.join(config['output_model_path'])


filenames = os.listdir(prod_directory)

# Check and read new data first, read ingestedfiles.txt
ingested_files = []
with open(os.path.join(prod_directory, 'ingestedfiles.txt')) as file:
    ingested_files = file.read().splitlines()


# Determine whether the source data folder has files that aren't listed in
# ingestedfiles.txt
source_files = os.listdir(input_path)

is_new = False
for new_file in source_files:
    is_new = new_file not in ingested_files
if not is_new:
    logging.info("No new files found")
    exit(0)

# Ingest new data
ingestion.merge_multiple_dataframe()

# Checking for model drift
with open(os.path.join(prod_directory, 'latestscore.txt')) as file:
    old_f1_score = float(file.read())

pred, y_test = model_predictions(output_folder_path, 'finaldata.csv')
new_f1_score = float(f1_score(pred, y_test))

# Check whether the score from the deployed model is different from the
# score from the model that uses the newest ingested data
if new_f1_score >= old_f1_score:
    logging.info("No model drift detected")
    exit(0)

# If there is model drift, retrain
logging.info("Retraining and redeploying the model")
train_model()

# Re-deploy if there is evidence for model drift, re-run the deployment.py
# script
deployment.store_model_into_pickle(artifacts_path)

# Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model
os.system("python3 reporting.py")
os.system("python3 apicalls.py")
