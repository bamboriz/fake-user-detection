from pathlib import Path
import warnings
warnings.filterwarnings(action='ignore')

from train import train_model
from inference import make_predictions

ROOT_DIR = Path('./')
DATA_DIR = ROOT_DIR / 'data'
MODELS_DIR = ROOT_DIR / 'models'

train_filepath = DATA_DIR / 'fake_users.csv'
test_filepath = DATA_DIR / 'fake_users_test.csv'
model_filepath = MODELS_DIR / 'model'

response = train_model(train_filepath, MODELS_DIR)
print(response)
response = make_predictions(test_filepath, model_filepath)
print(response)
