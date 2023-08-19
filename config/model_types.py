from sklearn.linear_model import ElasticNet, SGDRegressor, LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, VotingRegressor, StackingRegressor, GradientBoostingRegressor, ExtraTreesRegressor, BaggingRegressor, AdaBoostRegressor
from dataclasses import dataclass, field, make_dataclass
from typing import List, Union, Type
import sklearn.metrics as metrics
import sklearn
import inspect
from src.Exceptions.index import *







class ModelArchive():
    def __init__(self):



    #----------------------------------------------------------------
    #                         CHANGE HERE
    #----------------------------------------------------------------



        self.regression:dict=   {
                                'models':
                                    {
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
                                    },


                                'scoring':
                                    {
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
                                }
            
        
        self.classification:dict= {
                                        'models': {},
                                        'scoring': {}
                                    }
        

        self.clustering:dict= {}
        self.dimensionality_reduction:dict= {}
        self.scalers:dict= {}
        self.classification_scoring = {}

        #----------------------------------------------------------------
        #
        #----------------------------------------------------------------






@dataclass
class InquiryEngine:
    model_archive= ModelArchive()

    @property
    def regression(self):
        return self.model_archive.regression
    
    @property
    def classification(self):
        return self.model_archive.classification
    
    @property
    def all_available_models(self):
        return list(self.model_archive.regression['models'].values()) + list(self.model_archive.classification['models'].values())
    
    @property
    def all_available_model_names(self):
        return list(self.model_archive.regression.keys()) + list(self.model_archive.classification.keys())
    
    @property
    def all_available_scoring_name(self):
        return list(self.model_archive.regression['scoring'].keys()) + list(self.model_archive.classification['scoring'].keys())
    
    @property
    def all_available_scorings(self):
        return list(self.model_archive.regression['scoring'].values()) + list(self.model_archive.classification['scoring'].values())



@dataclass
class RegressionModelAssignment:
    model_type:str
    archive= ModelArchive()
    
    def validate(self):
        if self.model_type not in list(self.archive.regression['models'].keys()):
            raise InvalidModelException
    @property
    def model(self):
        self.validate()
        return self.archive.regression['models'][self.model_type]
    @property
    def hyperparameters(self):
        knowledge = {}
        signature = inspect.signature(self.model.__init__)
        for name, param in signature.parameters.items():
            #Iterating over parameters of a single model
            if (name != "self") & (param.default != inspect.Parameter.empty):
                knowledge[name] = {}
                knowledge[name]['type']= type(param.default)
                knowledge[name]['default'] = param.default
        return knowledge
    
    @property
    def hyperparameters_dataclass(self):
        fields = [(param, self.hyperparameters[param]['type'], field(default=self.hyperparameters[param]['default'])) for param in self.hyperparameters.keys()]
        data_class = make_dataclass(self.model_type, fields)
        return data_class
    
    @property
    def all_available_model_names(self):
        return list(self.archive.regression['models'].keys())
    
    @property
    def scoring_names(self):
        return list(self.archive.regression['scoring'])
    

@dataclass
class ClassificationModelAssignment:
    model_type:str
    model_archive= ModelArchive().classification

    @property
    def model(self):
        return self.model_archive[self.model_type]
    @property
    def hyperparameters(self):
        knowledge = {}
        signature = inspect.signature(self.model.__init__)
        for name, param in signature.parameters.items():
            #Iterating over parameters of a single model
            if (name != "self") & (param.default != inspect.Parameter.empty):
                knowledge[name] = {}
                knowledge[name]['type']= type(param.default)
                knowledge[name]['default'] = param.default
        return knowledge
    
    @property
    def hyperparameters_dataclass(self):
        fields = [(param, self.hyperparameters[param]['type'], field(default=self.hyperparameters[param]['default'])) for param in self.hyperparameters.keys()]
        data_class = make_dataclass(self.model_type, fields)
        return data_class
    
    @property
    def all_available_model_names(self):
        return list(self.model_archive.keys())
    
    @property
    def all_available_models(self):
        return list(self.model_archive.values())
    
    
    

    