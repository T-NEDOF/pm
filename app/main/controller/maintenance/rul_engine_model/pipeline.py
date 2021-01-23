from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor   
from app.main.controller.maintenance.rul_engine_model.processing import preprocessors as pp
from app.main.controller.maintenance.rul_engine_model.processing import features
from app.main.controller.maintenance.rul_engine_model.config import config

import logging


_logger = logging.getLogger(__name__)
# define pipelines
# random forest regression
reg_random_forest_pipeline = Pipeline(
    [
        ('drop_features',
            pp.DropUnecessaryFeatures(variables_to_drop=config.FEATURES_LOWCR)),
        ('reg_random_forest_model', RandomForestRegressor(max_depth=7, random_state=123))
    ]
)
# decision regression
reg_decision_tree_pipeline = Pipeline(
    [
        ('drop_features',
            pp.DropUnecessaryFeatures(variables_to_drop=config.FEATURES_LOWCR)),
        ('reg_decision_tree_model', DecisionTreeRegressor(max_depth=7, random_state=123))
    ]
)
# linear
reg_linear_regression_pipeline = Pipeline(
    [
        ('drop_features',
            pp.DropUnecessaryFeatures(variables_to_drop=config.FEATURES_LOWCR)),
        ('reg_linear_model', linear_model.LinearRegression())
    ]
)
# random forest classification
clf_random_forest_pipeline = Pipeline(
    [
        ('drop_features',
            pp.DropUnecessaryFeatures(variables_to_drop=config.FEATURES_LOWCR)),
        ('clf_random_forest_model',  RandomForestClassifier(n_estimators=50, random_state=123))
    ]
)
# decision tree classificaton
clf_decision_tree_pipeline = Pipeline(
    [
        ('drop_features',
            pp.DropUnecessaryFeatures(variables_to_drop=config.FEATURES_LOWCR)),
        ('clf_decision_tree_model', DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=5,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=123, splitter='best'))
    ]
)
# kneightbor
clf_knn_pipeline = Pipeline(
    [
        ('drop_features',
            pp.DropUnecessaryFeatures(variables_to_drop=config.FEATURES_LOWCR)),
        ('clf_knn_model', KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=-1, n_neighbors=13, p=2,
                     weights='uniform'))
    ]
)
# svm
clf_svc_pipeline = Pipeline(
    [
        ('drop_features',
            pp.DropUnecessaryFeatures(variables_to_drop=config.FEATURES_LOWCR)),
        ('clf_svc_model', SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
    kernel='rbf', max_iter=-1, probability=False, random_state=123,
    shrinking=True, tol=0.001, verbose=False))
    ]
)
# GaussianNB
clf_gauss_pipeline = Pipeline(
    [
        ('drop_features',
            pp.DropUnecessaryFeatures(variables_to_drop=config.FEATURES_LOWCR)),
        ('clf_gaussionNB_model', GaussianNB(priors=None, var_smoothing=1e-09))
    ]
)