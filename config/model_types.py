from sklearn.linear_model import ElasticNet, SGDRegressor, LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, VotingRegressor, StackingRegressor, GradientBoostingRegressor, ExtraTreesRegressor, BaggingRegressor, AdaBoostRegressor
from dataclasses import dataclass
from typing import Union, NewType
import sklearn.metrics as metrics

@dataclass
class ModelAssignment:
    #----------------------------------------------------------------
    #                         CHANGE HERE
    #----------------------------------------------------------------
    regression:dict= {
                'LinearRegression': LinearRegression,
                'SGDRegressor': SGDRegressor,
                'Lasso': Lasso,
                'ElasticNet': ElasticNet,
                'Ridge': Ridge,
                'SVR': SVR,
                'AdaboostRegressor': AdaBoostRegressor,
                'BaggingRegressor': BaggingRegressor,
                'HistGradientBoostingRegressor':  HistGradientBoostingRegressor,
                'VotingRegressor': VotingRegressor,
                'StackingRegressor': StackingRegressor,
                'GradientBoostingRegressor': GradientBoostingRegressor,
                'ExtraTreesRegressor':ExtraTreesRegressor,
                'XGBRegressor': XGBRegressor,
                'RandomForestRegressor': RandomForestRegressor
                }
    
    classification:dict= {}
    clustering:dict= {}
    dimensionality_reduction:dict= {}
    scalers:dict= {}

    regression_scoring = {
                            'explained_variance_score':metrics.explained_variance_score,
                            'max_error':metrics.max_error,
                            'neg_mean_absolute_error':metrics.mean_absolute_error,
                            'neg_mean_squared_error':metrics.mean_squared_error,
                            'neg_root_mean_squared_error':metrics.mean_squared_error,
                            'neg_mean_squared_log_error':metrics.mean_squared_log_error,
                            'neg_median_absolute_error':metrics.median_absolute_error,
                            'r2_score':metrics.r2_score,
                            'neg_mean_poisson_deviance':metrics.mean_poisson_deviance,
                            'neg_mean_gamma_deviance':metrics.mean_gamma_deviance,
                            'neg_mean_absolute_percentage_error':metrics.mean_absolute_percentage_error,
                            'd2_absolute_error_score':metrics.d2_absolute_error_score,
                            'd2_pinball_score':metrics.d2_pinball_score,
                            'd2_tweedie_score':metrics.d2_tweedie_score,
    }

    #----------------------------------------------------------------
    #
    #----------------------------------------------------------------


    RegressorModelType=  Union[*regression.values()]
    ClusteringModelType=  Union[*clustering.values()]
    ClassificationModelType=  Union[*classification.values()]