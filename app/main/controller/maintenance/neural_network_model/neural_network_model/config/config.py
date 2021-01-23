# The Keras model loading function does not play well with
# Pathlib at the moment, so we are using the old os module
# style

import os

PWD = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.abspath(os.path.join(PWD, '..'))
DATASET_DIR = os.path.join(PACKAGE_ROOT, 'datasets')
TRAINED_MODEL_DIR = os.path.join(PACKAGE_ROOT, 'trained_models')
DATA_FOLDER = os.path.join(DATASET_DIR, 'v2-plant-seedlings-dataset')

# MODEL PERSISTING
MODEL_NAME = 'lstm_model'
PIPELINE_NAME = 'cnn_pipe'
CLASSES_NAME = 'classes'
ENCODER_NAME = 'encoder'
MODEL_PATH = ""

# DATASET
TESTING_DATA_FILE = 'PM_test.txt'
TRAINING_DATA_FILE = 'PM_train.txt'
LTSM_DATA_FILE = 'LSTM_test.csv'

# MODEL
REGRESSION_MODEL = 'regression_model'
CLASSIFICATION_MODEL = 'binary_model'

# MODEL FITTING
IMAGE_SIZE = 150  # 50 for testing, 150 for final model
BATCH_SIZE = 10
EPOCHS = int(os.environ.get('EPOCHS', 1))  # 1 for testing, 10 for final model


with open(os.path.join(PACKAGE_ROOT, 'VERSION')) as version_file:
    _version = version_file.read().strip()

REGRESSION_MODEL_FILE_NAME = f'{REGRESSION_MODEL}.h5'
REGRESSION_MODEL_JSON_FILE_NAME = f'{REGRESSION_MODEL}.json'
REGRESSION_MODEL_PATH = os.path.join(DATASET_DIR, REGRESSION_MODEL_FILE_NAME)

CLASSIFICATION_MODEL_FILE_NAME = f'{CLASSIFICATION_MODEL}.h5'
CLASSIFICATION_MODEL_JSON_FILE_NAME = f'{REGRESSION_MODEL}.json'
CLASSIFICATION_MODEL_PATH = os.path.join(DATASET_DIR, CLASSIFICATION_MODEL_FILE_NAME)

PIPELINE_FILE_NAME = f'{PIPELINE_NAME}_{_version}.pkl'
PIPELINE_PATH = os.path.join(TRAINED_MODEL_DIR, PIPELINE_FILE_NAME)

CLASSES_FILE_NAME = f'{CLASSES_NAME}_{_version}.pkl'
CLASSES_PATH = os.path.join(TRAINED_MODEL_DIR, CLASSES_FILE_NAME)

ENCODER_FILE_NAME = f'{ENCODER_NAME}_{_version}.pkl'
ENCODER_PATH = os.path.join(TRAINED_MODEL_DIR, ENCODER_FILE_NAME)


COLUMS_MINMAXNORM = ['setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                     's15', 's16', 's17', 's18', 's19', 's20', 's21']