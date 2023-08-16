import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split  
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_percentage_error   
from sklearn.metrics import mean_absolute_error
from dataclasses import make_dataclass, field
import inspect
from ...config.model_types import ModelAssignment

from ..Exceptions.index import *

class Regression:

    def __init__(self, data:pd.DataFrame, target_column:str, train_size:float=0.6, test_size:float=0.2, datetime_column:str=None, random_state:int=0):

        if datetime_column== None:
            data = data.reset_index(drop=True).reset_index()
            datetime_column= 'index'
        self.data= data
        self.datetime_column = datetime_column
        self.target_column = target_column

        # Dropping nan values
        data= data.dropna()

        # Raising exception if data is not present
        if data.shape[0] == 0:
            raise NoDataPresentException
        
        # sorting values accoriding to datetime
        data = data.sort_values(by = self.datetime_column, ascending = True)
        self.data_test = data.tail(int(test_size*len(data)))

        data = data.drop(self.data_test.index)
        self.y = data[target_column]
        self.X = data.drop(target_column, axis = 1)
        self.X_test = self.data_test.drop(target_column, axis = 1)
        self.y_test = self.data_test[target_column]
        self.data_for_graph= {}

        self.train_size = train_size*len(self.X)        
        self.random_state = random_state
        
        self.x_train, self.x_valid, self.y_train, self.y_valid = train_test_split(self.X, self.y, train_size = self.train_size, random_state = self.random_state)
        self.x_valid1 = self.x_valid.drop(self.datetime_column, axis = 1)
        self.x_train1 = self.x_train.drop(self.datetime_column, axis = 1)

    
    @staticmethod
    def evaluate(test:pd.Series, preds: pd.Series, process:str)->tuple:
        score=r2_score(test,preds)
        mse = mean_squared_error(test,preds)
        rmse = np.sqrt(mean_squared_error(test,preds))
        mape = mean_absolute_percentage_error(test,preds)
        mae= mean_absolute_error(test, preds)

        print('For {}'.format(process))
        print('r2_score: {}\nmse:{}\nrmse:{}\nmape:{}\nmae:{}'.format(score, mse, rmse, mape, mae))

        return (score, mse, rmse, mape, mae)
    
    @staticmethod
    def get_all_configs()->dict:
        return ModelAssignment.regression
    
    @staticmethod
    def get_configs(model_types:list)->ModelAssignment.RegressorModelType:
        available_models= Regression.get_all_configs()
        models={}
        for model_type in model_types:
            if model_type in available_models.keys():
                models[model_type] = available_models[model_type]
            else:
                raise InvalidModelException
        return models
        
    @staticmethod
    def get_all_model_names()->list:
        return list(Regression.get_configs().keys())
    
    @staticmethod
    def get_all_models()->list:
        return list(Regression.get_configs().values())

    @staticmethod
    def get_extended_configs(model_types:list)->list:

        knowledge= {}
        model_assignment= Regression.get_configs(model_types)
        for model in model_assignment.keys():
            knowledge[model] = {}
            signature = inspect.signature(model_assignment[model].__init__)
            for name, param in signature.parameters.items():
                #Iterating over parameters of a single model
                if (name != "self") & (param.default != inspect.Parameter.empty):
                    knowledge[model][name] = {}
                    knowledge[model][name]['type']= type(param.default)
                    knowledge[model][name]['default'] = param.default
        return knowledge

    @staticmethod
    def get_dataclasses(model_types:list)->dict:

        data_classes= {}
        ext_configs= Regression.get_extended_configs(model_types)
        
        for model in ext_configs.keys():
            fields = [(param, ext_configs[model][param]['type'], field(default=ext_configs[model][param]['default'])) for param in ext_configs[model].keys()]
            data_classes[model] = make_dataclass(model, fields)
        
        return data_classes
    
    

