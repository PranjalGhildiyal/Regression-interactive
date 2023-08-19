import panel as pn
import pandas as pd
import re
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from src.Exceptions.index import *
from config.model_types import InquiryEngine
from src.Regression.Regression import Regression
from src.Regression.Regressors import Regressor


class Visualize_ML:
    def __init__(self, data:pd.DataFrame):


        # Cleaning the dataset
        self.data= Visualize_ML.cleaan_dataset(data)
        self.config_inquiry_engine= InquiryEngine()

        # Defining common widgets
        self.features_selector= pn.widgets.MultiChoice(name='Select Features', options= ['ALL'] + list(data.columns), value=['ALL'])
        self.datetime_column_selector= pn.widgets.Select(name= 'Select datetime column', options= [None] + list(data.columns), value= None)
        self.target_column_selector = pn.widgets.Select(name='Target Column', options=list(data.columns), value=list(data.columns)[-1])

        # Defining common variables
        self.model_types= {}                  # A dict to store the model types of different model type
        self.model_info= {}                   # A dict to store the model info in the form of dataclasses for that model type
        self.model_info_optimization = {}
        self.evaluate_buttons= {}             # A dict to store the evaluuate model buttns of different model tabs
        self.tab_num= 0                       # Initial tab number
        self.config_panel= {}                 # stores all the buttons
        self.tab_numbers= {}                  # A dict to store the tab numbers of different model type
        self.optimize_config_panel = {}
        self.optimize_evaluate_buttons= {}
        self.plotly_simple= {}
        self.plotly_optimization= {}
        self.model_objects = {}
        self.optimized_model_objects= {}
        self.train_size_selectors = {}
        self.train_size_selectors_optimization = {}
        self.random_state_selector = {}
        self.random_state_selector_optimization = {}
        self.gridspecs = {}
        self.gridspecs_optimization = {}
        self.model_specifics= {}                # Stores RegressionModelAssignment objects for different tabs.
        self.optimization_additionals= {}
        # self.plotly_widgets= {}
        # self.plotly_widgets_optimization= {}


    @staticmethod                   
    def cleaan_dataset(data:pd.DataFrame)->pd.DataFrame:
        all_columns = data.columns
        pattern = r'Unnamed: \d+|level_\d+|index'
        bad_columns = [col for col in all_columns if re.match(pattern, col)]

        return data.drop(bad_columns, axis = 1)
    
    @staticmethod
    def make_normal_options(widget, default, watcher_fnc):
        text_field=  pn.widgets.TextInput(name=widget.name, placeholder='defaults to {}'.format(default))
        text_field.param.watch(watcher_fnc, 'value')
        return pn.Row(widget,text_field)
    
    @staticmethod
    def change_button_color(widget,change_to='success'):
        widget.button_type=change_to
        


    

            


    # def evaluate_optimized(event, tab_num):



            
    # def change_text_fields(self, event=None):
    #     pass


    # # def change(self, event= None):

    # #     self.datetime_column= self.datetime_column_selector.value
    # #     if self.datetime_column != '':
    # #         self.data[self.datetime_column] = pd.to_datetime(self.data[self.datetime_column])
    # #         max_datetime = self.data[self.datetime_column].max()
    # #         min_datetime = self.data[self.datetime_column].min()
    # #         self.evaluate_buttons[self.tab_num]= pn.widgets.Button(name='Go', button_type= 'primary')
            
    # #         self.datetime_picker = pn.widgets.DatetimeRangePicker(value=(min_datetime, min_datetime),
    # #                                         start=min_datetime.date(), end=max_datetime.date(),
    # #                                         military_time=False, name='Datetime Range')
    # #     else:
    # #         self.datetime_picker= pn.Column()
        
    # #     self.model_types[self.tab_num]= self.model_type_selector.value
    # #     features = self.features_selector.value

    # #     #Defining dataclasses for this selection:
    # #     model_info= self.model_info[self.tab_num] = Regression.get_dataclasses([self.model_types[self.tab_num]])

    # #     self.make_fields(self.tab_num)

        





    #     for hyperparameter in model_info.keys():##########################################################################################################################



        
        






