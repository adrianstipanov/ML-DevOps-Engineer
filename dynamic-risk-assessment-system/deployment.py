"""
The module copies the model artifacts to a production deployment directory
"""

import logging
import os
import json
from shutil import copy2

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f)

# dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])
artifacts_path = os.path.join(config['output_model_path'])
ingestedfiles_path = os.path.join(config["output_folder_path"])


# function for deployment
def store_model_into_pickle(model):
    """ Copy the artifacts to prod deployment directory """
    for filename in os.listdir(model):
        copy2(
            os.path.join(
                artifacts_path, filename), os.path.join(
                prod_deployment_path, filename))

    copy2(
        os.path.join(
            ingestedfiles_path,
            'ingestedfiles.txt'),
        os.path.join(
            prod_deployment_path,
            'ingestedfiles.txt'))

    logging.info("Artifacts copied to deployment directory")


if __name__ == '__main__':
    store_model_into_pickle(artifacts_path)
