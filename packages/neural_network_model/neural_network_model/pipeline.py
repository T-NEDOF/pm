from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler

# from neural_network_model.config import config
from packages.neural_network_model.neural_network_model.config import config

# from neural_network_model.preprocessing import preprocessors as pp
from packages.neural_network_model.neural_network_model import model
from sklearn.compose import ColumnTransformer

clc_pipe = Pipeline([
                ('lstm_model_classification', model.lstm_clc)])

rgs_pipe = Pipeline([
                ('lstm_model_regression', model.lstm_rgs)])