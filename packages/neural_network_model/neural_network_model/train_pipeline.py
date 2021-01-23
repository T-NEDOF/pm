import joblib

from packages.neural_network_model.neural_network_model import pipeline
# from neural_network_model.config import config
from packages.neural_network_model.neural_network_model.config import config

from packages.neural_network_model.neural_network_model.processing import data_management as dm
from packages.neural_network_model.neural_network_model.processing import preprocessors as pp


def run_training(save_result: bool = True):

    train_data = dm.load_dataset(file_name=config.TRAINING_DATA_FILE)
    train_data = pp.preprocessing_train_data(train_data)
    x_train = pp.minmax_normalization_train(train_data)
    x = pp.GenSequence()
    x.fit(x_train)
    x_sequence_train = x.transform(x_train)

    y = pp.GenerateLabelsClassification()
    y.fit(train_data)
    label_array = y.transform(train_data)
    print(label_array)
    pipeline.clc_pipe.fit(x_sequence_train,label_array)
    pipeline.rgs_pipe.fit(x_sequence_train,label_array)

    if save_result:
        dm.save_pipeline_keras(pipeline.clc_pipe, config.CLASSIFICATION_MODEL_PATH,'lstm_model_classification')
        dm.save_pipeline_keras(pipeline.rgs_pipe, config.REGRESSION_MODEL_PATH,'lstm_model_regression')


if __name__ == '__main__':
    run_training(save_result=True)
