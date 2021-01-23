import numpy as np
from sklearn.model_selection import train_test_split

from app.main.controller.maintenance.rul_engine_model import pipeline
from app.main.controller.maintenance.rul_engine_model.processing.data_management import (load_dataset, save_pipeline)
from app.main.controller.maintenance.rul_engine_model.config import config
from app.main.controller.maintenance.rul_engine_model import __version__ as _version

import logging


_logger = logging.getLogger(__name__)


def run_training() -> None:
    """Train the model."""

    # read training data
    train_data = load_dataset(file_name=config.TRAINING_DATA_FILE)
    # test_data = load_dataset(file_name=config.TESTING_DATA_FILE)

    # divide train and test
    X_train = train_data[config.FEATURES_IMPOR]
    y_train_regression = train_data[config.TARGET_TTF]
    y_train_classification = train_data[config.TARGET_BNC]

    # train and save models in folder
    pipeline.reg_random_forest_pipeline.fit(X_train, y_train_regression)
    _logger.info(f'saving model version: {_version}')
    save_pipeline(pipeline_to_persist=pipeline.reg_random_forest_pipeline,model_name=config.REG_RF_PIPELINE_SAVE_FILE)

    pipeline.reg_decision_tree_pipeline.fit(X_train, y_train_regression)
    _logger.info(f'saving model version: {_version}')
    save_pipeline(pipeline_to_persist=pipeline.reg_decision_tree_pipeline,model_name=config.REG_DTR_PIPELINE_SAVE_FILE)

    pipeline.reg_linear_regression_pipeline.fit(X_train, y_train_regression)
    _logger.info(f'saving model version: {_version}')
    save_pipeline(pipeline_to_persist=pipeline.reg_linear_regression_pipeline
                                                        ,model_name=config.REG_LINEAR_REGRESSION_PIPELINE_SAVE_FILE)


    pipeline.clf_random_forest_pipeline.fit(X_train, y_train_classification)
    _logger.info(f'saving model version: {_version}')
    save_pipeline(pipeline_to_persist=pipeline.clf_random_forest_pipeline, model_name=config.CLF_RF_PIPELINE_SAVE_FILE)


    pipeline.clf_decision_tree_pipeline.fit(X_train, y_train_classification)
    _logger.info(f'saving model version: {_version}')
    save_pipeline(pipeline_to_persist=pipeline.clf_decision_tree_pipeline, model_name=config.CLF_DTR_PIPELINE_SAVE_FILE)


    pipeline.clf_knn_pipeline.fit(X_train, y_train_classification)
    _logger.info(f'saving model version: {_version}')
    save_pipeline(pipeline_to_persist=pipeline.clf_knn_pipeline, model_name=config.CLF_KNN_PIPELINE_SAVE_FILE)


    pipeline.clf_svc_pipeline.fit(X_train, y_train_classification)
    _logger.info(f'saving model version: {_version}')
    save_pipeline(pipeline_to_persist=pipeline.clf_svc_pipeline, model_name=config.CLF_SVC_PIPELINE_SAVE_FILE)

    pipeline.clf_gauss_pipeline.fit(X_train, y_train_classification)
    _logger.info(f'saving model version: {_version}')
    save_pipeline(pipeline_to_persist=pipeline.clf_gauss_pipeline, model_name=config.CLF_GAUSS_PIPELINE_SAVE_FILE)

if __name__ == '__main__':
    run_training()
