from sklearn.linear_model import ElasticNet, SGDRegressor, LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, VotingRegressor, StackingRegressor, GradientBoostingRegressor, ExtraTreesRegressor, BaggingRegressor, AdaBoostRegressor
from dataclasses import dataclass

@dataclass
class ModelAssignment:
    regression= {
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
    classification= {}
    clustering= {}
    dimensionality_reduction= {}
    scalers= {}





