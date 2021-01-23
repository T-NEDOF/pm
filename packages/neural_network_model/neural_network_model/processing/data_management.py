import logging
import os
import typing as t
from glob import glob
from pathlib import Path
import numpy as np

import pandas as pd
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from packages.neural_network_model.neural_network_model import model as m
# from neural_network_model.config import config
from packages.neural_network_model.neural_network_model.config import config


_logger = logging.getLogger(__name__)

def get_train_test_target(df: pd.DataFrame):
    """Split a dataset into train and test segments."""

    X_train, X_test, y_train, y_test = train_test_split(df['image'], df['target'], test_size=0.20, random_state=101)

    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)

    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return X_train, X_test, y_train, y_test


def save_pipeline_keras(model,MODEL_PATH,named_steps) -> None:
    """Persist keras model to disk."""
    model.named_steps[named_steps].model.save(MODEL_PATH)
    print("Saved model to disk")


def load_pipeline_keras() -> Pipeline:
    """Load a Keras Pipeline from disk."""
    build_model = lambda: load_model(config.CLASSIFICATION_MODEL_PATH)

    classifier = KerasClassifier(build_fn=build_model,
                          epochs=30,
                          batch_size=200,
                          validation_split=0.05,
                          verbose=2, 
                          callbacks=m.clc_callbacks_list
                          )

    classifier.model = build_model()

    return Pipeline([
        # ('dataset', dataset),
        ('lstm_model_classification', classifier)
    ])
    # return pipeline


def load_encoder() -> LabelEncoder:
    encoder = joblib.load(config.ENCODER_PATH)
    return encoder


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines, models, encoders and classes.

    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ['__init__.py']
    for model_file in Path(config.TRAINED_MODEL_DIR).iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()


def load_dataset(*, file_name: str
                 ) -> pd.DataFrame:
    _data = pd.read_csv(f'{config.DATASET_DIR}/{file_name}', sep=" ", header=None)
    return _data