import pandas as pd
import numpy as np
import mlflow
from sklearn.linear_model import ElasticNet, SGDRegressor, LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, VotingRegressor, StackingRegressor, GradientBoostingRegressor, ExtraTreesRegressor, BaggingRegressor, AdaBoostRegressor

from Regression import Regression


class Regressor(Regression):


    def __init__(self, 
                data:pd.DataFrame, 
                target_column:str,
                model_type:str='LinearRegression',
                train_size:float=0.6,
                test_size:float=0.2, 
                datetime_column:str=None,
                random_state:int=0,
                **kwargs):

        self.model_assignment= {
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
                                'XGBRegressor': XGBRegressor
                                }
        super().__init__(data, target_column, train_size, test_size, datetime_column, random_state)
        session = mlflow.start_run()
        
        with session :
            
            ## Building a model
            self.model =  self.model_assignment[model_type](**kwargs)#LinearRegression(**kwargs)
            self.model.fit(self.x_train1,self.y_train)
            
            ## On Training Data:
            y_pred_train = self.model.predict(self.x_train1)
            (self.train_score, self.train_mse, self.train_rmse, self.train_mape, self.train_mae) = Regression.evaluate(self.y_train, y_pred_train, 'Train')
            self.data_for_graph['Train'] = pd.DataFrame({'Actual': self.y_train, 'Predictions': y_pred_train, 'Date_time': self.x_train[self.datetime_column]}).sort_values(by = 'Date_time', ascending=True).reset_index(drop = True)

            ## On Validation Data:
            y_pred_valid = self.model.predict(self.x_valid1)
            (self.valid_score, self.valid_mse, self.valid_rmse, self.valid_mape, self.valid_mae) = Regression.evaluate(self.y_valid, y_pred_valid, 'Validation')
            self.data_for_graph['Validation'] = pd.DataFrame({'Actual': self.y_valid, 'Predictions': y_pred_valid, 'Date_time': self.x_valid[self.datetime_column]}).sort_values(by = 'Date_time', ascending=True).reset_index(drop = True)

            ## On Test data: 
            y_pred_test = self.model.predict(self.X_test.drop(self.datetime_column, axis = 1))
            (self.test_score, self.test_mse, self.test_rmse, self.test_mape, self.test_mae) = Regression.evaluate(self.y_test, y_pred_test, 'Test')
            self.data_for_graph['Test'] = pd.DataFrame({'Actual': self.y_test, 'Predictions': y_pred_test, 'Date_time': self.X_test[self.datetime_column]}).sort_values(by = 'Date_time', ascending=True).reset_index(drop = True)

            

            # Log parameters and metrics to MLflow
            mlflow.log_param("train_size", self.train_size)
            mlflow.log_param("random_state", self.random_state)
            mlflow.log_param("model_type", model_type)
            params= self.model.get_params()
            for param in params.keys():
                mlflow.log_params(param, params[param])

            mlflow.log_metric("r2_score_train", self.train_score)
            mlflow.log_metric("mse_train", self.train_mse)
            mlflow.log_metric("rmse_train", self.train_rmse)
            mlflow.log_metric("mape_train", self.train_mape)
            mlflow.log_metric("mae_train", self.train_mae)

            mlflow.log_metric("r2_score_valid", self.valid_score)
            mlflow.log_metric("mse_valid", self.valid_mse)
            mlflow.log_metric("rmse_valid", self.valid_rmse)
            mlflow.log_metric("mape_valid", self.valid_mape)
            mlflow.log_metric("mae_valid", self.valid_mae)

            mlflow.log_metric("r2_score_test", self.test_score)
            mlflow.log_metric("mse_test", self.test_mse)
            mlflow.log_metric("rmse_test", self.test_rmse)
            mlflow.log_metric("mape_test", self.test_mape)
            mlflow.log_metric("mae_test", self.train_mae)

            mlflow.sklearn.log_model(self.model, "model")
            
            return self

    def tune(**kwargs):
        def objective(trial):
            params=self.get_params()
            param= 
            n_estimators = trial.suggest_int('n_estimators', n_estimators_range[0], n_estimators_range[1])
            max_depth = int(trial.suggest_loguniform('max_depth', max_depth_range[0], max_depth_range[1]))
            regressor = RandomForestRegressor(n_estimators = n_estimators,  max_depth = max_depth)
            return np.absolute(sklearn.model_selection.cross_val_score(regressor, self.X, self.y, scoring=scoring, n_jobs=-1, cv=cv['cv'])).mean()