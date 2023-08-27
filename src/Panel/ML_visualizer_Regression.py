import panel as pn
import pandas as pd
import re
import math
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from src.Panel.ML_visualizer import Visualize_ML
from config.model_types import RegressionModelAssignment, ScoreInference
from src.Regression.Regressors import Regressor, Optimizer
from src.Regression.Regression import Regression
import subprocess


class Visualize_Regression(Visualize_ML):
    def __init__(self, data:pd.DataFrame):
        
        # Adding watcher functions for model type selector
        super().__init__(data)
        mlflow_url = """
                window.location.href="http://localhost:5000/"
                """
        my_github = """
                window.location.href="https://github.com/PranjalGhildiyal"
                """
        my_linkedin =  """
                window.location.href="https://www.linkedin.com/in/pranjal-ghildiyal/"
                """
        # self.config_inquiry_engine = self.config_inquiry_engine.regression
        self.config_inquiry_engine = self.config_inquiry_engine.regression
        self.model_type_selector= pn.widgets.Select(name='Select Regression Model', options= [''] + list(self.config_inquiry_engine['models'].keys()), value= '')
        self.score_inference = ScoreInference([0.2, 0.7])
        self.mlflow_button = pn.widgets.Button(name='MLflow', button_type='primary', sizing_mode='stretch_width')
        self.github_button = pn.widgets.Button(name='Github', button_type= 'default', sizing_mode= 'stretch_width', icon='robot')
        self.linkedin_button= pn.widgets.Button(name='LinekdIn', button_type= 'primary', sizing_mode= 'stretch_width', icon='progress')


        # Now adding watcher functions
        self.model_type_selector.param.watch(self.create_new_tab, 'value')
        subprocess.Popen(["mlflow", "ui"])
        self.mlflow_button.js_on_click(code=mlflow_url)
        self.github_button.js_on_click(code=my_github)
        self.linkedin_button.js_on_click(code=my_linkedin)
        self.band_error={}
        self.band_error_optimization= {}
        

        self.sidebar= pn.Column(
                                    self.model_type_selector,
                                    self.target_column_selector,
                                    self.features_selector,
                                    self.datetime_column_selector,
                                    self.mlflow_button,
                                    pn.Spacer(height= 375),
                                    pn.WidgetBox('# Follow me on:', pn.Row(self.github_button, self.linkedin_button), align='end')
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
        pn.serve(self.template)

        
    def create_new_tab(self, event):
        # Seaching if the model type is already present:
        if event.new not in self.tab_numbers.keys():
            print(event.new)
            model_type= self.model_types[self.tab_num]= self.model_type_selector.value
            self.tab_numbers[event.new] = self.tab_num

            print('Done initializing create_new_tab')

            #Defining dataclasses for this selection:
            self.model_specifics[self.tab_num] = RegressionModelAssignment(event.new)
            print('Creating mode info')
            self.model_info[self.tab_num] = self.model_specifics[self.tab_num].hyperparameters
            print('model info created successfully!')
            self.model_info_optimization[self.tab_num] = self.model_specifics[self.tab_num].hyperparameters

            # Creating initial plotly tab
            self.plotly_simple[self.tab_num] = pn.GridSpec(height= 1000)
            self.plotly_optimization[self.tab_num] = pn.GridSpec(height= 1000)

            #Initiating ScoreInferencing

            # Now creating buttons
            self.make_fields(self.tab_num)
            print('fields created successfully!')

            # Now that we have made the fields, we now make tabs

            # Binding both into one tab
            self.tab_model_type= pn.Tabs(
                                        ('Fit Model', pn.Row(self.config_panel[self.tab_num], self.plotly_simple[self.tab_num], sizing_mode='stretch_width')),
                                        ('Optimize Hyperparameters', pn.Row(self.optimize_config_panel[self.tab_num], self.plotly_optimization[self.tab_num], sizing_mode='stretch_width'))
                                    )
            self.tabs.append((model_type, self.tab_model_type))
            self.tabs.active = self.tab_num

            self.tab_num+=1

        else:
            self.tabs.active= self.tab_numbers[event.new]
        

    def make_fields(self, tab_num):
        print('in make_fields')
        self.random_state_selector[tab_num] = pn.widgets.FloatInput(name='Random State for train-test-split', value=0, start=0, end=100, step=1)
        self.train_size_selectors[tab_num] = pn.widgets.EditableRangeSlider(name='Range Slider', start=0, end=1, value=(0.6, 0.8), step=0.01)
        component= pn.Column('##### Train | Validation | Test', self.train_size_selectors[tab_num], self.random_state_selector[tab_num], sizing_mode='stretch_width',styles=dict(background='WhiteSmoke'))

        self.random_state_selector_optimization[tab_num] = pn.widgets.FloatInput(name='Random State for train-test-split', value=0, start=0, end=100, step=1)
        self.train_size_selectors_optimization[tab_num] = pn.widgets.EditableRangeSlider(name='Range Slider', start=0, end=1, value=(0.6, 0.8), step=0.01)

        component_optimized=  pn.Column('##### Train | Validation | Test', self.train_size_selectors_optimization[tab_num], self.random_state_selector_optimization[tab_num], sizing_mode='stretch_width', styles=dict(background='WhiteSmoke'))


        # Cross Validation and scoring metrics only for optimized
        self.optimization_additionals[tab_num] = pn.WidgetBox('## Cross Validation configs', sizing_mode='stretch_width')
        n_jobs= pn.widgets.IntInput(name='Parallel Jobs', start=0, end=10, step=1)
        n_splits= pn.widgets.IntInput(name='Splits', start=0, end=10, step=1)
        n_repeats= pn.widgets.IntInput(name='Repeats', start=0, end=15, step=1)
        random_state= pn.widgets.IntInput(name='Random State', start=0, end=10, step=1)
        scoring= pn.widgets.Select(name='Select Scoring method', options=list(self.config_inquiry_engine['scoring'].keys()), value='neg_mean_absolute_percentage_error')
        n_trials=pn.widgets.IntInput(name='No of Trials', value=100, start=0, end=1000, step=1)
        # In the next statement, the order matters
        self.optimization_additionals[tab_num].extend([n_jobs, n_splits, n_repeats, random_state, scoring, n_trials])

        

        self.config_panel[tab_num] = pn.Column(component, max_width= 350)
        self.optimize_config_panel[tab_num] = pn.Column(component_optimized, self.optimization_additionals[tab_num], max_width= 350)

        print('now making hyperparameter boxes')
        for hyperparameter in self.model_info[tab_num].keys():
            print(hyperparameter)
            default_value= self.model_info[tab_num][hyperparameter]['default']
            widget_box= pn.WidgetBox('### {}'.format(hyperparameter))
            print('\t{}'.format('now making'))


            if self.model_info[tab_num][hyperparameter]['type'] == bool:
                widget = pn.widgets.Toggle(name= hyperparameter, button_type= 'primary', value= default_value)
                widget.param.watch(lambda x: Visualize_ML.change_button_color(widget, change_to='success'), 'value')
                widget_box.append(widget)
                
            if self.model_info[tab_num][hyperparameter]['type'] == float:
                if math.isnan(default_value):
                    default_value=0
                widget = pn.widgets.EditableFloatSlider(name=hyperparameter, start=0, end=100, step=0.1, value=float(default_value))
                widget_box.append(widget)
                                                        
            if self.model_info[tab_num][hyperparameter]['type'] == int:
                widget= pn.widgets.EditableIntSlider(name=hyperparameter, start=0, end=100, step=1, value=int(default_value))
                widget_box.append(widget)

            if self.model_info[tab_num][hyperparameter]['type'] in [str, type(None)]:
                widget=  pn.widgets.TextInput(name=hyperparameter, placeholder='defaults to {}'.format(default_value))
                widget_box.append(widget)

            widget.param.watch(lambda event: self.change_hyperparameter(event, hyperparameter, tab_num), 'value')

            
            widget_optimizer = pn.widgets.TextInput(name=hyperparameter, placeholder= 'comma separated values or "-" separated range', value=None)
            widget_optimizer.param.watch(lambda event: self.change_optimizer_hyperparameter(event, hyperparameter, tab_num), 'value')

            widget.sizing_mode = 'stretch_width'
            widget_optimizer.sizing_mode = 'stretch_width'
            
            self.config_panel[tab_num].append(widget_box)
            self.optimize_config_panel[tab_num].append(widget_optimizer)

        # Now making the evaluate button:
        self.evaluate_buttons[tab_num] = pn.widgets.Button(name= 'Fit Model', button_type= 'primary', sizing_mode='stretch_width')
        self.evaluate_buttons[tab_num].on_click(lambda event: self.evaluate_normal(event, tab_num))
        self.config_panel[tab_num].append(self.evaluate_buttons[tab_num])

        self.optimize_evaluate_buttons[tab_num] = pn.widgets.Button(name= 'Optimize', button_type= 'primary')
        # disabling the evaluate button till a base-line model has been trained 
        self.optimize_evaluate_buttons[tab_num].disable = True
        self.optimize_evaluate_buttons[tab_num].on_click(lambda event: self.evaluate_optimized(event, tab_num))
        self.optimize_config_panel[tab_num].append(self.optimize_evaluate_buttons[tab_num])

    def change_hyperparameter(self, event, hyperparameter, tab_num):
        self.model_info[tab_num][hyperparameter]['default'] = event.new

    def change_optimizer_hyperparameter(self, event, hyperparameter, tab_num):

        # Now splitting this value
        event= event.replace(" ", "")
        list1= event.split(',')
        value = []
        for i in list1:
            list2= i.split('-')
            value.append(list2)
        self.model_info_optimization[tab_num][hyperparameter]['default'] = value



    def optimizer_enabler(self, tab_num):
        #Removing the evaluate button
        listt = self.optimize_config_panel[tab_num][0:-1]
        if (any(obj.value is not None for obj in listt)) & (tab_num in list(self.model_objects.keys())):
           self.optimize_evaluate_buttons[tab_num].disable= False
    
        
    def evaluate_normal(self, event, tab_num):

        hyperparameters= self.model_info[tab_num]

        selected_features= self.features_selector.value
        if 'ALL' in selected_features:
            selected_features = list(self.data.columns)
        if self.target_column_selector.value in self.features_selector.value:
            self.features_selector.value.remove(self.target_column_selector.value)

        selected_features = list(set(selected_features + [self.target_column_selector.value]))
        
        model_type= self.model_types[tab_num]
        new_data= self.data.filter(selected_features)
        target_column= self.target_column_selector.value
        train_size= self.train_size_selectors[tab_num].value[0]
        test_size= 1 - self.train_size_selectors[tab_num].value[1]
        datetime_column= self.datetime_column_selector.value
        random_state= self.random_state_selector[tab_num].value
        
        self.model_objects[tab_num] = Regressor(data=new_data, target_column=target_column, model_type=model_type, train_size= train_size, test_size= test_size, datetime_column=datetime_column, random_state=random_state, hyperparameters=hyperparameters).fit_model()

        train= self.model_objects[tab_num].data_for_graph['Train']
        validation= self.model_objects[tab_num].data_for_graph['Validation']
        test= self.model_objects[tab_num].data_for_graph['Test']

        self.make_plot_pane(tab_num, train, validation, test)
        self.tabs[tab_num][0][1]= self.gridspecs[tab_num]


    def evaluate_optimized(self, event, tab_num):
        
        hyperparameters = self.model_info_optimization[tab_num]

        
        # Specifying additional_models
        model_type= self.model_types[tab_num]
        new_data= self.data.filter(self.features_selector.value)
        target_column= self.target_column_selector.value
        train_size= self.train_size_selectors_optimization[tab_num].value[0]
        test_size= 1 - self.train_size_selectors_optimization[tab_num].value[1]
        random_state= self.random_state_selector_optimization[tab_num].value
        datetime_column= self.datetime_column_selector.value

        # Unzipping cross-validation and other button values
        cv = {'n_jobs': self.optimization_additionals[1].value,
              'n_splits': self.optimization_additionals[2].value,
              'n_repeats': self.optimization_additionals[3].value,
              'random_state': self.optimization_additionals[4].value
              }
        scoring = self.optimization_additionals[5].value
        n_trials = self.optimization_additionals[6].value


        self.optimized_model_objects[tab_num] = Optimizer(data=new_data, target_column=target_column, model_type=model_type, train_size= train_size, test_size= test_size, datetime_column=datetime_column, random_state=random_state, cv= cv, scoring=scoring, n_trials=n_trials, hyperparameters = hyperparameters).optimize()

        train= self.optimized_model_objects[tab_num].data_for_graph['Train']
        validation= self.optimized_model_objects[tab_num].data_for_graph['Validation']
        test= self.optimized_model_objects[tab_num].data_for_graph['Test']

        self.make_plot_pane_optimization(tab_num, train, validation, test, tab_num)
        self.tabs[tab_num][1][1]= self.gridspecs_optimization[tab_num]


    def make_plot_pane(self, tab_num:int, train:pd.DataFrame=None, validation:pd.DataFrame=None, test:pd.DataFrame=None):
        if ((train is None) | (test is None) | (validation is None)):
            self.gridspecs[tab_num] = pn.GridSpec()
            return 
        self.gridspecs[tab_num]= pn.GridSpec(sizing_mode='stretch_width', mode='warn',height= 1000)

        self.gridspecs[tab_num][0, 0:5] = self.plot_lineplot_only(df= train, process='Training Data')
        self.gridspecs[tab_num][0, 5] = self.make_statsbox(train, tab_num)

        self.gridspecs[tab_num][1, 0:5] = self.plot_lineplot_only(df= validation, process= 'Validation Data')
        self.gridspecs[tab_num][1, 5] = self.make_statsbox(validation, tab_num)

        widget= pn.widgets.EditableFloatSlider(name='Enter Tolerence (%)', start=0, end=10, step=0.1, value=0, sizing_mode='stretch_width')
        widget.param.watch(lambda event: self.change_lines(event, tab_num), 'value')

        self.gridspecs[tab_num][2, 0:5] = self.plot_lineplot(df=test, value= widget.value, tab_num=tab_num, process= 'Testing Data')
        self.gridspecs[tab_num][2, 5] = self.make_statsbox(test, tab_num)

        widget1 = pn.widgets.Button(name='Band Error | {}'.format(round(self.band_error[tab_num], 1)), button_type= self.score_inference.button_type(self.band_error[tab_num], 'Band Error'), sizing_mode='stretch_width')
        widget1.on_click(lambda event: self.update_band_error(widget1, tab_num))

        new_widgetbox= pn.WidgetBox()
        metric_options = [None] + self.model_specifics[tab_num].scoring_names
        metric_options= list(set(metric_options) - set(['r2_score', 'neg_root_mean_squared_error', 'neg_mean_squared_error', 'neg_mean_absolute_error']))
        self.metric_selector[tab_num]= pn.widgets.Select(name= 'Select Additional Metric', options= metric_options, value=None, sizing_mode='stretch_width')
        self.metric_selector[tab_num].param.watch(lambda event: self.add_metric(event, tab_num), 'value')

        new_widgetbox.extend([pn.Row(widget, widget1, self.metric_selector[tab_num])])

        self.gridspecs[tab_num][3, :]= new_widgetbox

    def make_plot_pane_optimization(self, tab_num:int, train:pd.DataFrame=None, validation:pd.DataFrame=None, test:pd.DataFrame=None):
        if ((train is None) | (test is None) | (validation is None)):
            self.gridspecs_optimization[tab_num] = pn.GridSpec()
            return 
        self.gridspecs_optimization[tab_num]= pn.GridSpec(sizing_mode='stretch_width', mode='warn',height= 1000)

        self.gridspecs_optimization[tab_num][0, 0:5] = self.plot_lineplot_only(df= train, process='Training Data')
        self.gridspecs_optimization[tab_num][0, 5] = self.make_statsbox(train, tab_num, model='optimization')

        self.gridspecs_optimization[tab_num][1, 0:5] = self.plot_lineplot_only(df= validation, process='Validation Data')
        self.gridspecs_optimization[tab_num][1, 5] = self.make_statsbox(validation, tab_num, model='optimization')

        widget= pn.widgets.EditableFloatSlider(name='Enter Tolerence (%)', start=0, end=10, step=0.1, value=0, sizing_mode='stretch_width')
        widget.param.watch(lambda event: self.change_lines_optimization(event, tab_num), 'value')

        self.gridspecs_optimization[tab_num][2, 0:5] = self.plot_lineplot_optimization(df=test, value= widget.value, tab_num=tab_num, process='Testing Data')
        self.gridspecs_optimization[tab_num][2, 5] = self.make_statsbox(test, tab_num, model='optimization')

        widget1 = pn.widgets.Button(name='Band Error | {}'.format(round(self.band_error_optimization[tab_num], 1)), button_type= self.score_inference.button_type(self.band_error_optimization[tab_num], 'Band Error'), sizing_mode='stretch_width')
        widget1.on_click(lambda event: self.update_band_error_optimization(widget1, tab_num))

        new_widgetbox= pn.WidgetBox()
        metric_options = [None] + self.model_specifics[tab_num].scoring_names
        metric_options= list(set(metric_options) - set(['r2_score', 'neg_root_mean_squared_error', 'neg_mean_squared_error', 'neg_mean_absolute_error']))
        self.metric_selector_optimization[tab_num]= pn.widgets.Select(name= 'Select Additional Metric', options= metric_options, value=None, sizing_mode='stretch_width')
        self.metric_selector_optimization[tab_num].param.watch(lambda event: self.add_metric(event, tab_num), 'value')
        new_widgetbox.extend([widget, self.metric_selector_optimization[tab_num]])

        self.gridspecs_optimization[tab_num][3, :] = new_widgetbox


    def update_band_error_optimization(self, widget, tab_num):
        widget.name = 'Band Error | {}'.format(round(self.band_error_optimization[tab_num], 1))
    
    def update_band_error(self, widget, tab_num):
        widget.name = 'Band Error | {}'.format(round(self.band_error[tab_num], 1))
    
    def change_lines(self, event, tab_num):
        self.tabs[tab_num][0][1][2, 0:5] = self.plot_lineplot(self.model_objects[tab_num].data_for_graph['Test'], value= event.new, tab_num=tab_num, process='Testing Data')

    def change_lines_optimization(self, event, tab_num):
        self.tabs[tab_num][1][1][2, 0:5] = self.plot_lineplot_optimization(self.optimized_model_objects[tab_num].data_for_graph['Test'], value= event.new, tab_num=tab_num, process='Testing Data')
    
    def add_metric(self, event, tab_num):
        if event.new:
            metric= self.model_specifics[tab_num].scoring[event.new]
            train=  self.model_objects[tab_num].data_for_graph['Train']
            test= self.model_objects[tab_num].data_for_graph['Test']
            validation = self.model_objects[tab_num].data_for_graph['Validation']

            train_value= round(metric(train['Actual'], train['Predictions']), 1)
            test_value= round(metric(test['Actual'], test['Predictions']), 1)
            validation_value= round(metric(validation['Actual'], validation['Predictions']), 1)
            
            train_button= pn.widgets.Button(name= '{} | {}'.format(event.new, train_value), button_type= self.score_inference.button_type(train_value, event.new), sizing_mode= 'stretch_width', button_style='outline')
            test_button= pn.widgets.Button(name= '{} | {}'.format(event.new, test_value), button_type= self.score_inference.button_type(test_value, event.new), sizing_mode= 'stretch_width', button_style='outline')
            validation_button= pn.widgets.Button(name= '{} | {}'.format(event.new, validation_value), button_type= self.score_inference.button_type(validation_value, event.new), sizing_mode= 'stretch_width', button_style='outline')

            self.tabs[tab_num][0][1][0, 5].append(train_button)
            self.tabs[tab_num][0][1][1, 5].append(validation_button)
            self.tabs[tab_num][0][1][2, 5].append(test_button)

            options= self.metric_selector[tab_num].options
            options= list(set(options) - set([event.new]))

            self.metric_selector[tab_num].options=options
            self.metric_selector[tab_num].value= None

    def add_metric_optimization(self, event, tab_num):
        if event.new:
            metric= self.model_specifics[tab_num].scoring[event.new]
            train=  self.model_objects[tab_num].data_for_graph['Train']
            test= self.model_objects[tab_num].data_for_graph['Test']
            validation = self.model_objects[tab_num].data_for_graph['Validation']

            train_value= round(metric(train['Actual'], train['Predictions']), 1)
            test_value= round(metric(test['Actual'], test['Predictions']), 1)
            validation_value= round(metric(validation['Actual'], validation['Predictions']), 1)
            
            train_button= pn.widgets.Button(name= '{} | {}'.format(event.new, train_value), button_type= self.score_inference.button_type(train_value, event.new), sizing_mode= 'stretch_width', button_style='outline')
            test_button= pn.widgets.Button(name= '{} | {}'.format(event.new, test_value), button_type= self.score_inference.button_type(test_value, event.new), sizing_mode= 'stretch_width', button_style='outline')
            validation_button= pn.widgets.Button(name= '{} | {}'.format(event, validation_value), button_type= self.score_inference.button_type(validation_value, event.new), sizing_mode= 'stretch_width', button_style='outline')

            self.tabs[tab_num][1][1][0, 5].append(train_button)
            self.tabs[tab_num][1][1][1, 5].append(validation_button)
            self.tabs[tab_num][1][1][2, 5].append(test_button)

            options= self.metric_selector_optimization[tab_num].options
            options= list(set(options) - set([event.new]))

            self.metric_selector_optimization[tab_num].options = options
            self.metric_selector_optimization[tab_num].value=None


        
    def plot_lineplot(self, df, value, tab_num, event=None, process:str='Testing Data'):

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Making boundaries
        df['+Actual'] = (1 + value) * df['Actual']
        df['-Actual'] = (1 - value) * df['Actual']

        df['+Predictions'] = (1 + value) * df['Predictions']
        df['-Predictions'] = (1 - value) * df['Predictions']

        # Shade Area
        fig.add_trace(go.Scatter(x=df.index, y=df['+Actual'], fill= None, mode='lines', line_color='rgba(0,0,255,0.2)', showlegend=False), secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df['-Actual'], fill='tonexty', mode='lines', line_color='rgba(0,0,255,0.2)', showlegend=False), secondary_y=False)

        # Shade Area
        fig.add_trace(go.Scatter(x=df.index, y=df['+Predictions'], fill= None, mode='lines', line_color='rgba(255,165,0,255)', showlegend=False), secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df['-Predictions'], fill='tonexty', mode='lines', line_color='rgba(255,165,0,255)', showlegend=False), secondary_y=False)

        # Add traces
        fig.add_trace(go.Scatter(x=df.index, y=df['Actual'], name='Actual Values', line=dict(color='blue')), secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df['Predictions'], name='Predictions', line=dict(color='red')), secondary_y=False)

        fig.update_layout(height= 400, title_text=process)

        # # Set titles
        # fig.update_yaxes(title_text="<b>primary</b>", secondary_y=False)
        # fig.update_yaxes(title_text="<b>secondary</b>", secondary_y=True)

        df['error'] = np.nan
        mask= df['Predictions'] > df['+Actual']
        if df[mask].shape[0] > 0:
            df.loc[mask, 'error'] = (df.loc[mask, 'Predictions'] - df.loc[mask, '+Actual'])/df.loc[mask, '+Actual']

        mask= df['Predictions'].between(df['-Actual'],df['+Actual'])
        if df[mask].shape[0] > 0:
            df.loc[mask, 'error'] = 0

        mask= df['Predictions'] < df['-Actual']
        if df[mask].shape[0] > 0:
            df.loc[mask, 'error'] = (df.loc[mask, '-Actual'] - df.loc[mask, 'Predictions'])/df.loc[mask, '-Actual']

        self.band_error[tab_num] = df['error'].mean()

        return pn.pane.Plotly(fig, sizing_mode='stretch_both')
    
    def plot_lineplot_optimization(self, df, value, tab_num, event=None, process:str='Testing Data'):

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Making boundaries
        df['+Actual'] = (1 + value) * df['Actual']
        df['-Actual'] = (1 - value) * df['Actual']

        df['+Predictions'] = (1 + value) * df['Predictions']
        df['-Predictions'] = (1 - value) * df['Predictions']

        # Shade Area
        fig.add_trace(go.Scatter(x=df.index, y=df['+Actual'], fill= None, mode='lines', line_color='rgba(0,0,255,0.2)', showlegend=False), secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df['-Actual'], fill='tonexty', mode='lines', line_color='rgba(0,0,255,0.2)', showlegend=False), secondary_y=False)

        # Shade Area
        fig.add_trace(go.Scatter(x=df.index, y=df['+Predictions'], fill= None, mode='lines', line_color='rgba(255,165,0,255)', showlegend=False), secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df['-Predictions'], fill='tonexty', mode='lines', line_color='rgba(255,165,0,255)', showlegend=False), secondary_y=False)

        # Add traces
        fig.add_trace(go.Scatter(x=df.index, y=df['Actual'], name='Actual Values', line=dict(color='blue')), secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df['Predictions'], name='Predictions', line=dict(color='red')), secondary_y=False)

        fig.update_layout(height= 400, title_text=process)

        # # Set titles
        # fig.update_yaxes(title_text="<b>primary</b>", secondary_y=False)
        # fig.update_yaxes(title_text="<b>secondary</b>", secondary_y=True)

        df['error'] = np.nan
        mask= df['Predictions'] > df['+Actual']
        if df[mask].shape[0] > 0:
            df.loc[mask, 'error'] = (df.loc[mask, 'Predictions'] - df.loc[mask, '+Actual'])/df.loc[mask, '+Actual']

        mask= df['Predictions'].between(df['-Actual'],df['+Actual'])
        if df[mask].shape[0] > 0:
            df.loc[mask, 'error'] = 0

        mask= df['Predictions'] < df['-Actual']
        if df[mask].shape[0] > 0:
            df.loc[mask, 'error'] = (df.loc[mask, '-Actual'] - df.loc[mask, 'Predictions'])/df.loc[mask, '-Actual']

        self.band_error_optimization[tab_num] = df['error'].mean()

        return pn.pane.Plotly(fig, sizing_mode='stretch_both')
    
    def plot_lineplot_only(self, df, process, event=None):

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        fig.add_trace(go.Scatter(x=df.index, y=df['Actual'], name='Actual Values', line=dict(color='blue')), secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df['Predictions'], name='Predictions', line=dict(color='red')), secondary_y=False)

        # # Set titles
        # fig.update_yaxes(title_text="<b>primary</b>", secondary_y=False)
        # fig.update_yaxes(title_text="<b>secondary</b>", secondary_y=True)

        fig.update_layout(height= 400, title_text=process)

        return pn.pane.Plotly(fig, sizing_mode='stretch_both')

    def make_statsbox(self, data, tab_num, model=None):
        score, mse, rmse, mape, mae = Regression.evaluate(data['Actual'],data['Predictions'])

        output= pn.Column(scroll=True)

        score_button= pn.widgets.Button(name= 'R2 Score | {}'.format(round(score, 1)), button_type= self.score_inference.button_type(score, 'score'), sizing_mode= 'stretch_width', button_style='outline')
        mse_button= pn.widgets.Button(name= 'MSE | {}'.format(round(mse, 1)), button_type= self.score_inference.button_type(mse, 'error'), sizing_mode= 'stretch_width', button_style='outline')
        rmse_button= pn.widgets.Button(name= 'RMSE | {}'.format(round(rmse, 1)), button_type= self.score_inference.button_type(rmse, 'error'), sizing_mode= 'stretch_width', button_style='outline')
        mape_button= pn.widgets.Button(name= 'MAPE | {}'.format(round(mape, 1)), button_type= self.score_inference.button_type(mape, 'error'), sizing_mode= 'stretch_width', button_style='outline')
        mae_button= pn.widgets.Button(name= 'MAE | {}'.format(round(mae, 1)), button_type= self.score_inference.button_type(mae, 'error'), sizing_mode= 'stretch_width', button_style='outline')

        if model:
            self.band_error_optimization[tab_num] = mape
        else:
            self.band_error[tab_num] = mape

        output.extend([score_button, mse_button, rmse_button, mape_button, mae_button])
        return output
