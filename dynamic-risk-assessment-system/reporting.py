from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json
import os
import logging
from diagnostics import model_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path'])


y_pred, y_test = model_predictions()


# Function for reporting
def score_model():
    cm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.savefig(os.path.join(output_model_path, "confusionmatrix.png"))

    logging.info("Saved confusion matrix")


if __name__ == '__main__':
    score_model()
