import numpy as np
from sklearn.model_selection import train_test_split

from app.main.controller.maintenance.rul_engine_model import pipeline
from app.main.controller.maintenance.rul_engine_model.processing.data_management import (
    load_dataset, save_pipeline)
from app.main.controller.maintenance.rul_engine_model.config import config
from app.main.controller.maintenance.rul_engine_model import __version__ as _version

import logging


_logger = logging.getLogger(__name__)


def run_training() -> None:
    """Train the model."""

    # read training data
    train_data = load_dataset(file_name=config.TRAINING_DATA_FILE)
    test_data = load_dataset(file_name=config.TESTING_DATA_FILE)

    # divide train and test
    X_train = train_data[config.FEATURES_IMPOR]
    y_train = train_data[config.TARGET]

    X_test = test_data[config.FEATURES_IMPOR]
    y_test = test_data[config.TARGET]

    # transform the target
    # y_train = np.log(y_train)
    # y_test = np.log(y_test)

    pipeline.random_forest_pipeline.fit(X_train, y_train)

    _logger.info(f'saving model version: {_version}')
    save_pipeline(pipeline_to_persist=pipeline.random_forest_pipeline)


if __name__ == '__main__':
    run_training()
