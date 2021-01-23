import numpy as np
import pandas as pd

from app.main.controller.maintenance.rul_engine_model.processing.data_management import load_pipeline
from app.main.controller.maintenance.rul_engine_model.config import config
from app.main.controller.maintenance.rul_engine_model.processing.validation import validate_inputs
from app.main.controller.maintenance.rul_engine_model import __version__ as _version

import logging
import typing as t


_logger = logging.getLogger(__name__)
# Load models
reg_rf_pipeline_file_name = f'{config.REG_RF_PIPELINE_SAVE_FILE}{_version}.pkl'
_reg_rf_pipe = load_pipeline(file_name=reg_rf_pipeline_file_name)

reg_dtr_pipeline_file_name = f'{config.REG_DTR_PIPELINE_SAVE_FILE}{_version}.pkl'
_reg_dtr_pipe = load_pipeline(file_name=reg_dtr_pipeline_file_name)

reg_linear_pipeline_file_name = f'{config.REG_LINEAR_REGRESSION_PIPELINE_SAVE_FILE}{_version}.pkl'
_reg_linear_pipe = load_pipeline(file_name=reg_linear_pipeline_file_name)



clf_rf_pipeline_file_name = f'{config.CLF_RF_PIPELINE_SAVE_FILE}{_version}.pkl'
_clf_random_forest_pipe = load_pipeline(file_name=clf_rf_pipeline_file_name)

clf_dtr_pipeline_file_name = f'{config.CLF_DTR_PIPELINE_SAVE_FILE}{_version}.pkl'
_clf_decidion_tree_pipe = load_pipeline(file_name=clf_dtr_pipeline_file_name)

clf_knn_pipeline_file_name = f'{config.CLF_KNN_PIPELINE_SAVE_FILE}{_version}.pkl'
_clf_knn_pipe = load_pipeline(file_name=clf_knn_pipeline_file_name)

clf_svc_pipeline_file_name = f'{config.CLF_SVC_PIPELINE_SAVE_FILE}{_version}.pkl'
_clf_svc_pipe = load_pipeline(file_name=clf_svc_pipeline_file_name)

clf_gauss_pipeline_file_name = f'{config.CLF_GAUSS_PIPELINE_SAVE_FILE}{_version}.pkl'
_clf_gauss_pipe = load_pipeline(file_name=clf_gauss_pipeline_file_name)

def make_prediction_by_reg_random_forest(*, input_data: t.Union[pd.DataFrame, dict],
                    ) -> dict:
    """Make a prediction using a saved model pipeline.
    Args:
        input_data: Array of model prediction inputs.
    Returns:
        Predictions for each input row, as well as the model version.
    """
    data = pd.DataFrame(input_data)
    validated_data = validate_inputs(input_data=data)
    prediction = _reg_rf_pipe.predict(validated_data[config.FEATURES_IMPOR])
    # output = np.exp(prediction)
    results = {'predictions': prediction, 'version': _version}
    _logger.info(
        f'Making predictions with model version: {_version} '
        f'Inputs: {validated_data} '
        f'Predictions: {results}')
    return results

def make_prediction_by_reg_decision_tree(*, input_data: t.Union[pd.DataFrame, dict],
                    ) -> dict:
    """Make a prediction using a saved model pipeline.
    Args:
        input_data: Array of model prediction inputs.
    Returns:
        Predictions for each input row, as well as the model version.
    """
    data = pd.DataFrame(input_data)
    validated_data = validate_inputs(input_data=data)
    prediction = _reg_dtr_pipe.predict(validated_data[config.FEATURES_IMPOR])
    # output = np.exp(prediction)
    results = {'predictions': prediction, 'version': _version}
    _logger.info(
        f'Making predictions with model version: {_version} '
        f'Inputs: {validated_data} '
        f'Predictions: {results}')
    return results

def make_prediction_by_reg_linear_regression(*, input_data: t.Union[pd.DataFrame, dict],
                    ) -> dict:
    """Make a prediction using a saved model pipeline.
    Args:
        input_data: Array of model prediction inputs.
    Returns:
        Predictions for each input row, as well as the model version.
    """
    data = pd.DataFrame(input_data)
    validated_data = validate_inputs(input_data=data)
    prediction = _reg_linear_pipe.predict(validated_data[config.FEATURES_IMPOR])
    # output = np.exp(prediction)
    results = {'predictions': prediction, 'version': _version}
    _logger.info(
        f'Making predictions with model version: {_version} '
        f'Inputs: {validated_data} '
        f'Predictions: {results}')
    return results

def make_prediction_by_clf_random_forest(*, input_data: t.Union[pd.DataFrame, dict],
                    ) -> dict:
    """Make a prediction using a saved model pipeline.

    Args:
        input_data: Array of model prediction inputs.

    Returns:
        Predictions for each input row, as well as the model version.
    """
    data = pd.DataFrame(input_data)
 
    validated_data = validate_inputs(input_data=data)

    prediction = _clf_random_forest_pipe.predict(validated_data[config.FEATURES_IMPOR])

    # output = np.exp(prediction)

    results = {'predictions': prediction, 'version': _version}

    _logger.info(
        f'Making predictions with model version: {_version} '
        f'Inputs: {validated_data} '
        f'Predictions: {results}')

    return results

def make_prediction_by_clf_decision_tree(*, input_data: t.Union[pd.DataFrame, dict],
                    ) -> dict:
    """Make a prediction using a saved model pipeline.

    Args:
        input_data: Array of model prediction inputs.

    Returns:
        Predictions for each input row, as well as the model version.
    """
    data = pd.DataFrame(input_data)
 
    validated_data = validate_inputs(input_data=data)

    prediction = _clf_decidion_tree_pipe.predict(validated_data[config.FEATURES_IMPOR])

    # output = np.exp(prediction)

    results = {'predictions': prediction, 'version': _version}

    _logger.info(
        f'Making predictions with model version: {_version} '
        f'Inputs: {validated_data} '
        f'Predictions: {results}')

    return results

def make_prediction_by_clf_knn(*, input_data: t.Union[pd.DataFrame, dict],
                    ) -> dict:
    """Make a prediction using a saved model pipeline.

    Args:
        input_data: Array of model prediction inputs.

    Returns:
        Predictions for each input row, as well as the model version.
    """
    data = pd.DataFrame(input_data)
 
    validated_data = validate_inputs(input_data=data)

    prediction = _clf_knn_pipe.predict(validated_data[config.FEATURES_IMPOR])

    # output = np.exp(prediction)

    results = {'predictions': prediction, 'version': _version}

    _logger.info(
        f'Making predictions with model version: {_version} '
        f'Inputs: {validated_data} '
        f'Predictions: {results}')

    return results

def make_prediction_by_clf_svc(*, input_data: t.Union[pd.DataFrame, dict],
                    ) -> dict:
    """Make a prediction using a saved model pipeline.

    Args:
        input_data: Array of model prediction inputs.

    Returns:
        Predictions for each input row, as well as the model version.
    """
    data = pd.DataFrame(input_data)

    validated_data = validate_inputs(input_data=data)

    prediction = _clf_svc_pipe.predict(validated_data[config.FEATURES_IMPOR])

    # output = np.exp(prediction)

    results = {'predictions': prediction, 'version': _version}

    _logger.info(
        f'Making predictions with model version: {_version} '
        f'Inputs: {validated_data} '
        f'Predictions: {results}')

    return results

def make_prediction_by_clf_gauss(*, input_data: t.Union[pd.DataFrame, dict],
                    ) -> dict:
    """Make a prediction using a saved model pipeline.

    Args:
        input_data: Array of model prediction inputs.

    Returns:
        Predictions for each input row, as well as the model version.
    """
    data = pd.DataFrame(input_data)
 
    validated_data = validate_inputs(input_data=data)

    prediction = _clf_gauss_pipe.predict(validated_data[config.FEATURES_IMPOR])

    # output = np.exp(prediction)

    results = {'predictions': prediction, 'version': _version}

    _logger.info(
        f'Making predictions with model version: {_version} '
        f'Inputs: {validated_data} '
        f'Predictions: {results}')

    return results