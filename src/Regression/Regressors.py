import pandas as pd
import numpy as np
import sklearn
import mlflow
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold 
import re
from src.Regression.Regression import Regression
from config.model_types import RegressionModelAssignment


class Regressor(Regression):


    def __init__(self, 
                data:pd.DataFrame, 
                target_column:str,
                model_type:str='LinearRegression',
                train_size:float=0.6,
                test_size:float=0.2, 
                datetime_column:str=None,
                random_state:int=0,
                hyperparameters:dict={}):

        super().__init__(data, target_column, train_size, test_size, datetime_column, random_state)
        self.model=RegressionModelAssignment(model_type).model
        self.model_type= model_type
        

        self.hyperparameters_new = {}
        for hyperparameter in hyperparameters:
            self.hyperparameters_new[hyperparameter] = hyperparameters[hyperparameter]['default']

    def fit_model(self)->'Regressor':
        mlflow.set_tracking_uri('')
        session = mlflow.start_run()
        
        with session :
            
            ## Building a model
            self.model = self.model(**self.hyperparameters_new)
            self.model.fit(self.x_train,self.y_train)
            
            ## On Training Data:
            y_pred_train = self.model.predict(self.x_train)
            (self.train_score, self.train_mse, self.train_rmse, self.train_mape, self.train_mae) = Regression.evaluate(self.y_train, y_pred_train, 'Train')
            self.data_for_graph['Train'] = pd.DataFrame({'Actual': self.y_train, 'Predictions': y_pred_train, 'Date_time': self.x_train.index}).sort_values(by = 'Date_time', ascending=True).reset_index(drop = True)

            ## On Validation Data:
            y_pred_valid = self.model.predict(self.x_valid)
            (self.valid_score, self.valid_mse, self.valid_rmse, self.valid_mape, self.valid_mae) = Regression.evaluate(self.y_valid, y_pred_valid, 'Validation')
            self.data_for_graph['Validation'] = pd.DataFrame({'Actual': self.y_valid, 'Predictions': y_pred_valid, 'Date_time': self.x_valid.index}).sort_values(by = 'Date_time', ascending=True).reset_index(drop = True)

            ## On Test data: 
            y_pred_test = self.model.predict(self.X_test)
            (self.test_score, self.test_mse, self.test_rmse, self.test_mape, self.test_mae) = Regression.evaluate(self.y_test, y_pred_test, 'Test')
            self.data_for_graph['Test'] = pd.DataFrame({'Actual': self.y_test, 'Predictions': y_pred_test, 'Date_time': self.X_test.index}).sort_values(by = 'Date_time', ascending=True).reset_index(drop = True)

            

            # Log parameters and metrics to MLflow
            mlflow.log_param("train_size", self.train_size)
            mlflow.log_param("random_state_train_test_split", self.random_state)
            mlflow.log_param("model_type", self.model_type)
            params= self.model.get_params()
            mlflow.log_params(params)

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
        
class Optimizer(Regression):


    def __init__(self, 
                data:pd.DataFrame, 
                target_column:str,
                scoring:str,
                cv:dict,
                model_type:str='LinearRegression',
                train_size:float=0.6,
                test_size:float=0.2, 
                datetime_column:str=None,
                random_state:int=0,
                buttons:dict=None,
                n_trials:int=100,
                **hyperparameters):
        
        super().__init__(data, target_column, train_size, test_size, datetime_column, random_state)
        self.model=RegressionModelAssignment(model_type).model
        self.hyperparameters= hyperparameters
        self.scoring= scoring
        self.n_trials= n_trials
        self.cv= cv
        self.model_type= model_type

    def optimize(self)->'Optimizer':

        def objective(trial):
            trial_hyperparameters= {}
            for i in self.hyperparameters:
                type= self.hyperparameters[i]['type']
                value=self.hyperparameters[i]['default']
                result = []
                for item in value:
                    if isinstance(item, list):
                        result.extend(range(item[0], item[1] + 1))
                    else:
                        result.append(item)
                result= list(set(result))

                hyperparameter_range= [max(result), min(result)]

                if (type == type(None)) or (type == str):
                    type = Optimizer.classify_math_number(value)


                if type == type(int):
                    trial_hyperparameters[i] = trial.suggest_int(i, hyperparameter_range[0], hyperparameter_range[1])
                if type == type(float):
                    trial_hyperparameters[i] = trial.suggest_float(i, hyperparameter_range[0], hyperparameter_range[1])
                if type == type(str):
                    pass

            regressor = self.model(**trial_hyperparameters)
            return np.absolute(sklearn.model_selection.cross_val_score(regressor, self.X, self.y, scoring=self.scoring, n_jobs=-1, cv=self.cv['n_splits'])).mean()

        # OPTUNA APPLICATION
        if 'score' in self.scoring:
            study = optuna.create_study(direction='maximize')
        else:
            study = optuna.create_study(direction='minimize')

        study.optimize(objective, n_trials=self.n_trials)
        trial = study.best_trial
        self.optimization_traectory= optuna.visualization.plot_optimization_history(study)

        # Random Forest: final best model   
        mlflow.set_tracking_uri('')
        session = mlflow.start_run()
        with session :
            # Building the model
            self.model = self.model(**trial.params)

            # Cross Validation
            CV = RepeatedKFold(n_splits=self.cv['n_splits'], n_repeats=self.cv['n_repeats'], random_state=self.cv['random_state'])
            scores = cross_val_score(self.model, self.X, self.y, scoring=self.scoring, cv=CV, n_jobs=self.cv['n_jobs'])
            scores = np.absolute(scores)
            mlflow.log_metric("Cross Validation: mean_{}".format(self.scoring), scores.mean())
            mlflow.log_metric("Cross Validation: std_{}".format(self.scoring), scores.std())
            

            # Model fitting
            self.model.fit(self.x_train,self.y_train)
            
            ## On Training Data:
            y_pred_train = self.model.predict(self.x_train)
            (self.train_score, self.train_mse, self.train_rmse, self.train_mape, self.train_mae) = Regression.evaluate(self.y_train, y_pred_train, 'Train')
            self.data_for_graph['Train'] = pd.DataFrame({'Actual': self.y_train, 'Predictions': y_pred_train, 'Date_time': self.x_train.index}).sort_values(by = 'Date_time', ascending=True).reset_index(drop = True)

            ## On Validation Data:
            y_pred_valid = self.model.predict(self.x_valid)
            (self.valid_score, self.valid_mse, self.valid_rmse, self.valid_mape, self.valid_mae) = Regression.evaluate(self.y_valid, y_pred_valid, 'Validation')
            self.data_for_graph['Validation'] = pd.DataFrame({'Actual': self.y_valid, 'Predictions': y_pred_valid, 'Date_time': self.x_valid.index}).sort_values(by = 'Date_time', ascending=True).reset_index(drop = True)

            ## On Test data: 
            y_pred_test = self.model.predict(self.X_test)
            (self.test_score, self.test_mse, self.test_rmse, self.test_mape, self.test_mae) = Regression.evaluate(self.y_test, y_pred_test, 'Test')
            self.data_for_graph['Test'] = pd.DataFrame({'Actual': self.y_test, 'Predictions': y_pred_test, 'Date_time': self.X_test.index}).sort_values(by = 'Date_time', ascending=True).reset_index(drop = True)


            # Log parameters and metrics to MLflow
            mlflow.log_param("train_size", self.train_size)
            mlflow.log_param("random_state_train_test_split", self.random_state)
            mlflow.log_param("model_type", self.model_type)
            mlflow.log_params(trial.params)

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

            return self 
    
    @staticmethod
    def classify_math_number(input_str):
        if re.match(r'^[+-]?\d+$', input_str):
            return type(int)
        elif re.match(r'^[+-]?\d+\.\d*$', input_str):
            return type(float)
        else:
            return type(str)
