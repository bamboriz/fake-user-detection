import pandas as pd
from pathlib import Path

from pycaret.classification import setup, create_model, save_model, load_model, predict_model
from pycaret.utils import check_metric

from preprocess import preprocess

def train_model(training_data_filepath: str, models_dir: Path) -> dict:
    """
    training_data_filepath (str): the path to the training data
    :returns (dict) {"model_performance": "", "model_path": ""} 
    """
    X = pd.read_csv(training_data_filepath)
    X_preprocessed = preprocess(X)
    data, data_unseen = split_data(X_preprocessed)
    model_path = train_job(data, models_dir)
    score = evaluate_model(data_unseen, model_path)
    return {"model_performance": score, "model_path": model_path}

def split_data(X: pd.DataFrame) -> tuple:
    train = X.sample(frac=0.95, random_state=786)
    test = X.drop(train.index)
    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)
    print('Data for Modeling: ' + str(train.shape))
    print('Unseen Data For Predictions: ' + str(test.shape))
    return train, test

def train_job(X: pd.DataFrame, models_dir: Path) -> str:
    predictors = list(X.columns).remove('fake')
    exp_clf101 = setup(data = X, target = 'fake', session_id=1, numeric_features = predictors, ignore_features = ['user'], silent=True)
    knn = create_model('knn', fold = 5)
    model_filepath = models_dir / 'model'
    _ = save_model(knn, model_filepath)
    return model_filepath

def evaluate_model(X: pd.DataFrame, model_path: Path) -> float:
    model = load_model(model_path)
    new_prediction = predict_model(model, data=X)
    score = check_metric(new_prediction['fake'], new_prediction['Label'], metric = 'Accuracy')
    return score