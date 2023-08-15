import panel as pn
import pandas as pd
import re
from ..Exceptions import *
from ...config.model_types import ModelAssignment
from ..Regression.Regression import Regression


class Visualize_ML:
    def __init__(self, data:pd.DataFrame):


        # Cleaning the dataset
        self.data= Visualize_ML.cleaan_dataset(data)

        # Defining common widgets
        self.model_type_selector= pn.widgets.Select(name='Select Regression Model', options= [''] + list(ModelAssignment.regression.keys()), value= '')
        self.features_selector= pn.widgets.MultiChoice(name='Select Features', options= ['ALL'] + list(data.columns), value=['ALL'])
        self.datetime_column_selector= pn.widgets.Select(name= 'Select datetime column', options= [''] + list(data.columns), value= '')
        
        


    @staticmethod                   
    def cleaan_dataset(data:pd.DataFrame)->pd.DataFrame:
        all_columns = data.columns
        pattern = r'Unnamed: \d+|level_\d+|index'
        bad_columns = [col for col in all_columns if re.match(pattern, col)]

        return data.drop(bad_columns, axis = 1)
    
    @staticmethod
    def make_options(widget, default, watcher_func):
        text_field=  pn.widgets.TextInput(name=widget.name, placeholder='defaults to {}'.format(default))
        text_field.param.watch(watcher_func, 'value')
        return pn.Row(widget,text_field)
    
    

class Visualize_Regression(Visualize_ML):
    def __init__(self, data:pd.DataFrame):
        super().__init__(data)
        self.model_types= {}
        self.model_info= {}
        self.evaluate_buttons= {}
        self.tab_num= 0
        self.config_panel= {}


    def make_fields(self, tab_num):
        self.config_panel[tab_num] = pn.Column()
        for hyperparameter in self.model_info[tab_num].keys():
            default_value= hyperparameter['default']


            if self.model_info[tab_num][hyperparameter] == type(bool):
                widget = pn.widgets.Button(name= hyperparameter, button_type= 'primary')
                
            if self.model_info[tab_num][hyperparameter] == type(float):
                widget = pn.widgets.EditableFloatSlider(name=hyperparameter, start=0, end=100, step=0.1, value=float(default_value))
                                                        
            if self.model_info[tab_num][hyperparameter] == type(int):
                widget= pn.widgets.EditableIntSlider(name=hyperparameter, start=0, end=100, step=1, value=int(default_value))

            else:
                widget=pn.Column()

            widget.param.watch(self.change, 'value')

            self.config_panel[tab_num].append(Visualize_ML.make_options(widget, default_value))
            # self.tab_buttons[tab_num][hyperparameter].onclick(lambda x: )

            
    def change_text_fields(self, event=None):
        pass


    def change(self, event= None):

        self.datetime_column= self.datetime_column_selector.value
        if self.datetime_column != '':
            self.data[self.datetime_column] = pd.to_datetime(self.data[self.datetime_column])
            max_datetime = self.data[self.datetime_column].max()
            min_datetime = self.data[self.datetime_column].min()
            self.evaluate_buttons[self.tab_num]= pn.widgets.Button(name='Go', button_type= 'primary')
            
            self.datetime_picker = pn.widgets.DatetimeRangePicker(value=(min_datetime, min_datetime),
                                            start=min_datetime.date(), end=max_datetime.date(),
                                            military_time=False, name='Datetime Range')
        else:
            self.datetime_picker= pn.Column()
        
        self.model_types[self.tab_num]= self.model_type_selector.value
        features = self.features_selector.value

        #Defining dataclasses for this selection:
        model_info= self.model_info[self.tab_num] = Regression.get_dataclasses([self.model_types[self.tab_num]])

        self.make_fields(self.tab_num)

        





        for hyperparameter in model_info.keys():##########################################################################################################################



        
        






