"""
This module includes unit tests for churn_library.py

Author: Adrian Stipanov
Date: Apr 16, 2022
"""
from pathlib import Path
import logging
import pandas as pd
import pytest
import churn_library as cl

DATA_PATH = "./data/bank_data.csv"

cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category']

# Logging config
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

# FIXTURES


@pytest.fixture(scope="module", name='path')
def path():
    """
    Fixture - The test function test_import_data() will
    use the return of path() as an argument
    """
    yield DATA_PATH


@pytest.fixture(name='data_frame')
def data_frame():
    """
    Fixture - The test function test_eda() and test_encoder_helper will
    use the return of data_frame() as an argument
    """
    yield cl.import_data(DATA_PATH)


@pytest.fixture(name='encoded_data_frame')
def encoded_data_frame(data_frame):
    """
    Fixture - The test function test_perform_feature_engineering() will
    use the return of encoded_data_frame() as an argument
    """
    yield cl.encoder_helper(data_frame, cat_columns)


@pytest.fixture(name='train_test_split')
def train_test_split(encoded_data_frame):
    """
    Fixture - The test function train_models() will
    use the return of train_test_split() as an argument
    """
    return cl.perform_feature_engineering(encoded_data_frame)


# UNIT TESTS

def test_import_data(path):
    '''
    test import data from csv
    '''

    try:
        data_frame = cl.import_data(path)

    except FileNotFoundError as err:
        logging.error("File not found")
        raise err

    # Check the df shape
    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0

    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return data_frame


def test_eda(data_frame):
    """
    Check if EDA results are saved
    """
    cl.perform_eda(data_frame)

    # Check if each file exists
    path = Path("./images/eda")

    for file in [
        "Churn",
        "Customer_Age",
        "Marital_Status",
        "Total_Trans_Ct",
            "heatmap"]:
        file_path = path.joinpath(f'{file}_distribution.png')
        try:
            assert file_path.is_file()
        except AssertionError as err:
            logging.error("ERROR: Eda results not found.")
            raise err
    logging.info("SUCCESS: EDA results successfully saved!")


def test_encoder_helper(data_frame):
    '''
    test encoder helper
    '''

    # Check if df is empty
    assert isinstance(data_frame, pd.DataFrame)
    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "ERROR: The dataframe doesn't appear to have rows and columns")
        raise err

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    data = cl.encoder_helper(data_frame, cat_columns)

    # Check if categorical columns exist in df
    try:
        for col in cat_columns:
            assert col in data_frame.columns
    except AssertionError as err:
        logging.error("ERROR: Missing categorical columns")
        raise err
    logging.info("SUCCESS: Categorical columns correctly encoded.")

    return data


def test_perform_feature_engineering(encoded_data_frame):
    '''
    test perform_feature_engineering
    '''

    X_train, X_test, y_train, y_test = cl.perform_feature_engineering(
        encoded_data_frame)

    try:
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
    except AssertionError as err:
        logging.error("ERROR: The shape of train test splits don't match")
        raise err
    logging.info("SUCCESS: Train test correctly split.")

    return (X_train, X_test, y_train, y_test)


def test_train_models(train_test_split):
    '''
    test train_models
    '''

    X_train, X_test, y_train, y_test = train_test_split

    # Train model
    cl.train_models(X_train, X_test, y_train, y_test)

    # Check if model were saved after done training
    path = Path("./models")

    models = ['logistic_model.pkl', 'rfc_model.pkl']

    for model_name in models:
        model_path = path.joinpath(model_name)
        try:
            assert model_path.is_file()
        except AssertionError as err:
            logging.error("ERROR: Models not found.")
            raise err
    logging.info("SUCCESS: Models successfully saved!")


if __name__ == "__main__":
    data_frame = test_import_data(DATA_PATH)
    test_eda(data_frame)
    encoded_data = test_encoder_helper(data_frame)
    features = test_perform_feature_engineering(encoded_data)
    test_train_models(features)
