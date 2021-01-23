from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from app.main.controller.maintenance.rul_engine_model.processing import preprocessors as pp
from app.main.controller.maintenance.rul_engine_model.processing import features
from app.main.controller.maintenance.rul_engine_model.config import config

import logging


_logger = logging.getLogger(__name__)

random_forest_pipeline = Pipeline(
    [
        ('drop_features',
            pp.DropUnecessaryFeatures(variables_to_drop=config.FEATURES_LOWCR)),
        ('random_forest_model', RandomForestRegressor(n_estimators=100, max_features=3, max_depth=4, n_jobs=-1, random_state=1))
    ]
)
