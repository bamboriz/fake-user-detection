import pandas as pd
import numpy as np
from pathlib import Path

from pycaret.classification import load_model, predict_model
from pycaret.utils import check_metric

from preprocess import preprocess


def make_predictions(data_filepath: str, models_path: Path) -> np.ndarray:
    """
    data_df (pd.DataFrame): a pandas dataframe with the inference data
    data_filepath (str): the path to the inference data
    :returns (np.ndarray) the model predictions (output of the model.predict() method)
    """
    X = pd.read_csv(data_filepath)
    X_preprocessed = preprocess(X)
    X_preprocessed.pop('fake') # not necessary bc predict model does not use this, but just want to show that the target col is not used for predicting 
    model = load_model(models_path)
    predictions = predict_model(model, data=X_preprocessed)
    return predictions['Label'].values