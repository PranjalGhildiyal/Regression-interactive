import panel as pn
import pandas as pd
import re
from ..Exceptions import *
from ...config.model_types import ModelAssignment
from ..Regression.Regression import Regression
from ..Regression.Regressors import Regressor


class Visualize_ML:
    def __init__(self, data:pd.DataFrame):


        # Cleaning the dataset
        self.data= Visualize_ML.cleaan_dataset(data)

        # Defining common widgets
        self.features_selector= pn.widgets.MultiChoice(name='Select Features', options= ['ALL'] + list(data.columns), value=['ALL'])
        self.datetime_column_selector= pn.widgets.Select(name= 'Select datetime column', options= [''] + list(data.columns), value= '')
        self.target_column_selector = pn.widgets.Select(name='Target Column', options=list(data.columns), value=list(data.columns)[-1])

        # Defining common variables
        self.model_types= {}                  # A dict to store the model types of different model type
        self.model_info= {}                   # A dict to store the model info in the form of dataclasses for that model type
        self.evaluate_buttons= {}             # A dict to store the evaluuate model buttns of different model tabs
        self.tab_num= 0                       # Initial tab number
        self.config_panel= {}                 # stores all the buttons
        self.tab_numbers= {}                  # A dict to store the tab numbers of different model type
        self.optimize_config_panel = {}
        self.optimize_evaluate_buttons= {}
        self.plotly_simple= {}
        self.plotly_optimization= {}
        self.model_objects = {}
        self.train_size_selectors = {}
        self.train_size_selectors_optimization = {}
        self.random_state_selector = {}
        self.random_state_selector_optimization = {}


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
        

class Visualize_Regression(Visualize_ML):
    def __init__(self, data:pd.DataFrame):
        self.model_type_selector= pn.widgets.Select(name='Select Regression Model', options= [''] + list(ModelAssignment.regression.keys()), value= '')
        # adding watcher functions for model type selector

        super().__init__(data)

        self.sidebar= pn.Column(
                                    self.model_type_selector,
                                    self.features_selector,
                                    self.datetime_column_selector
                                )
        
        self.tabs = pn.Tabs()

        self.template = pn.template.BootstrapTemplate(
                                                        title="Regressor",
                                                        sidebar=self.sidebar,
                                                        header_color = '#FFFFFF',
                                                        background_color = '#FFFFFF',
                                                        corner_radius = 1,
                                                        main_layout = ''
                                                    )
        self.template.main.append(self.tabs)
        
        
    def create_new_tab(self, event):
        # Seaching if the model type is already present:
        if event.new not in self.tab_numbers.keys():
            model_type= self.model_types[self.tab_num]= self.model_type_selector.value
            self.tab_numbers[event.new] = self.tab_num

            # Creating initial plotly tab
            self.plotly_simple[self.tab_num] = self.plot_plotly_pane()
            self.plotly_optimization[self.tab_num] = self.plot_plotly_pane()

            # Now creating buttons
            self.make_fields(self.tab_num)

            # Now that we have made the fields, we now make tabs
            ##  Making simple page
            simple_page= pn.Column(self.config_panel[self.tab_num], self.plotly_simple[self.tab_num])

            ##  Making Optimized page
            optimized_page= pn.Column(self.optimize_config_panel, self.plotly_optimization[self.tab_num])

            # Binding both into one tab
            tab_model_type= pn.Tabs(
                                        (model_type, simple_page),
                                        ('Optimize Hyperparameters', optimized_page)
                                    )
            self.tabs.append(tab_model_type)
            self.tabs.active = self.tab_num

            self.tab_num+=1

        else:
            self.tabs.active= self.tab_numbers[event.new]
        

    def make_fields(self, tab_num):

        self.random_state_selector[tab_num] = pn.widgets.FloatInput(name='Random State for train-test-split', value=0, start=0, end=100, step=1)
        self.train_size_selectors[tab_num] = pn.widgets.EditableRangeSlider(name='Range Slider', start=0, end=1, value=(0.6, 0.8), step=0.01)
        component= pn.Column('##### Train | Validation | Test', pn.Row(self.train_size_selectors[tab_num], self.random_state_selector[tab_num]))

        self.random_state_selector_optimization[tab_num] = pn.widgets.FloatInput(name='Random State for train-test-split', value=0, start=0, end=100, step=1)
        self.train_size_selectors_optimization[tab_num] = pn.widgets.EditableRangeSlider(name='Range Slider', start=0, end=1, value=(0.6, 0.8), step=0.01)
        component_optimized=  pn.Column('##### Train | Validation | Test', pn.Row(self.train_size_selectors_optimization[tab_num], self.random_state_selector_optimization[tab_num]))

        self.config_panel[tab_num] = pn.Column(component, max_width= 100)
        self.optimize_config_panel[tab_num] = pn.Column(component_optimized, max_width= 100)

        for hyperparameter in self.model_info[tab_num].keys():

            default_value= hyperparameter['default']
            widget_box= pn.WidgetBox('### {}'.format(hyperparameter))


            if self.model_info[tab_num][hyperparameter] == type(bool):
                widget = pn.widgets.Toggle(name= hyperparameter, button_type= 'primary', value= default_value)
                widget.param.watch(lambda x: Visualize_ML.change_button_color(widget, change_to='success'))
                widget_box.append(widget)
                
            if self.model_info[tab_num][hyperparameter] == type(float):
                widget = pn.widgets.EditableFloatSlider(name=hyperparameter, start=0, end=100, step=0.1, value=float(default_value))
                widget1=pn.widgets.FloatInput(name=hyperparameter, start=0, end=100000, step=0.1, value=float(default_value))
                widget.jslink(widget1, value='value', bidirectional=True)
                widget_box.extend([widget, widget1])
                                                        
            if self.model_info[tab_num][hyperparameter] == type(int):
                widget= pn.widgets.EditableIntSlider(name=hyperparameter, start=0, end=100, step=1, value=int(default_value))
                widget1= pn.widgets.IntInput(name=hyperparameter, start=0, end=10000, step=1, value=int(default_value))
                widget.jslink(widget1, value='value', bidirectional=True)
                widget_box.extend([widget, widget1])

            if self.model_info[tab_num][hyperparameter] == type(str):
                widget=  pn.widgets.TextInput(name=hyperparameter, placeholder='defaults to {}'.format(default_value))
                widget_box.append(widget)

            if self.model_info[tab_num][hyperparameter] == type(None):
                widget=  pn.widgets.TextInput(name=hyperparameter, placeholder='defaults to {}'.format(default_value))
                widget_box.append(widget)

            
            widget_optimizer = pn.widgets.TextInput(name=hyperparameter, placeholder= 'comma separated values or "-" separated range')
            
            self.config_panel[tab_num].append(widget_box)
            self.optimize_config_panel[tab_num].append(widget_optimizer)

        # Now making the evaluate button:
        self.evaluate_buttons[tab_num] = pn.widgets.Button(name= 'Fit Model', button_type= 'primary')
        self.evaluate_buttons[tab_num].onclick(lambda event: self.evaluate_normal(event, tab_num))
        self.config_panel.append(self.evaluate_buttons[tab_num])

        self.optimize_evaluate_buttons[tab_num] = pn.widgets.Button(name= 'Optimize', button_type= 'primary')
        # disabling the evaluate button till a base-line model has been trained 
        self.optimize_evaluate_buttons[tab_num].disable = True
        self.optimize_evaluate_buttons[tab_num].onclick(lambda event: self.evaluate_optimized(event, tab_num))
        self.optimize_config_panel.append(self.optimize_evaluate_buttons[tab_num])


    def evaluate_normal(self, event, tab_num):
        hyperparameters = {}
        for hyperparameter in self.config_panel[tab_num].keys():
            name= self.config_panel[tab_num][hyperparameter][0].name
            value= self.config_panel[tab_num][hyperparameter][0].value
            hyperparameters[name] = value

        if self.target_column_selector.value in self.features_selector.value:
            self.features_selector.value.remove(self.target_column_selector.value)
        
        model_type= self.model_types[tab_num]
        new_data= self.data.filter(self.features_selector.value)
        target_column= self.target_column_selector.value
        train_size= self.train_size_selectors[tab_num][0]
        test_size= 1 - self.train_size_selectors[tab_num][0]
        datetime_column= self.datetime_column_selector.value
        random_state= self.random_state_selector[tab_num]
        
        self.model_objects[tab_num] = Regressor(data=new_data, target_column=target_column, model_type=model_type, train_size= train_size, test_size= test_size, datetime_column=datetime_column, random_state=random_state, **hyperparameters)

        train= self.model_objects[tab_num].data_for_graph['Train']
        validation= self.model_objects[tab_num].data_for_graph['Validation']
        test= self.model_objects[tab_num].data_for_graph['Test']
        
        self.plotly_simple[tab_num]= self.make_plot_pane(train, validation, test)













    def make_plot_pane(self, train:pd.DataFrame=None, validation:pd.DataFrame=None, test:pd.DataFrame=None);
        if not all([train, test, validation]):
            return pn.pane.Plotly('# We are still waiting for a response!')
        
    

        
            





            


    def evaluate_optimized(event, tab_num):



            
    def change_text_fields(self, event=None):
        pass


    # def change(self, event= None):

    #     self.datetime_column= self.datetime_column_selector.value
    #     if self.datetime_column != '':
    #         self.data[self.datetime_column] = pd.to_datetime(self.data[self.datetime_column])
    #         max_datetime = self.data[self.datetime_column].max()
    #         min_datetime = self.data[self.datetime_column].min()
    #         self.evaluate_buttons[self.tab_num]= pn.widgets.Button(name='Go', button_type= 'primary')
            
    #         self.datetime_picker = pn.widgets.DatetimeRangePicker(value=(min_datetime, min_datetime),
    #                                         start=min_datetime.date(), end=max_datetime.date(),
    #                                         military_time=False, name='Datetime Range')
    #     else:
    #         self.datetime_picker= pn.Column()
        
    #     self.model_types[self.tab_num]= self.model_type_selector.value
    #     features = self.features_selector.value

    #     #Defining dataclasses for this selection:
    #     model_info= self.model_info[self.tab_num] = Regression.get_dataclasses([self.model_types[self.tab_num]])

    #     self.make_fields(self.tab_num)

        





        for hyperparameter in model_info.keys():##########################################################################################################################



        
        






