import panel as pn
import pandas as pd
import re
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from src.Panel.ML_visualizer import Visualize_ML
from config.model_types import RegressionModelAssignment
from src.Regression.Regressors import Regressor
from src.Regression.Regression import Regression


class Visualize_Regression(Visualize_ML):
    def __init__(self, data:pd.DataFrame):
        
        # Adding watcher functions for model type selector
        super().__init__(data)
        # self.config_inquiry_engine = self.config_inquiry_engine.regression
        self.config_inquiry_engine = self.config_inquiry_engine.regression
        self.model_type_selector= pn.widgets.Select(name='Select Regression Model', options= [''] + list(self.config_inquiry_engine['models'].keys()), value= '')
        

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
        pn.serve(self.template)
        
        
    def create_new_tab(self, event):
        # Seaching if the model type is already present:
        if event.new not in self.tab_numbers.keys():
            model_type= self.model_types[self.tab_num]= self.model_type_selector.value
            self.tab_numbers[event.new] = self.tab_num

            #Defining dataclasses for this selection:
            self.model_specifics[self.tab_num] = RegressionModelAssignment(event.new)
            self.model_info[self.tab_num] = self.model_specifics[self.tab_num].hyperparameters

            # Creating initial plotly tab
            self.plotly_simple[self.tab_num] = self.make_plot_pane()
            self.plotly_optimization[self.tab_num] = self.make_plot_pane()

            # Now creating buttons
            self.make_fields(self.tab_num)

            # Now that we have made the fields, we now make tabs
            ##  Making simple page
            simple_page= pn.Column(self.config_panel[self.tab_num], self.plotly_simple[self.tab_num])

            ##  Making Optimized page
            optimized_page= pn.Column(self.optimize_config_panel[self.tab_num], self.plotly_optimization[self.tab_num])

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
        widget_box_optimized= pn.WidgetBox()

        component_optimized=  pn.Column('##### Train | Validation | Test', pn.Row(self.train_size_selectors_optimization[tab_num], self.random_state_selector_optimization[tab_num]))


        # Cross Validation and scoring metrics only for optimized
        self.optimization_additionals[tab_num] = pn.WidgetBox('## Cross Validation configs')
        n_jobs= pn.widgets.IntInput(name='Parallel Jobs', start=0, end=10, step=1)
        n_splits= pn.widgets.IntInput(name='Splits', start=0, end=10, step=1)
        n_repeats= pn.widgets.IntInput(name='Repeats', start=0, end=15, step=1)
        random_state= pn.widgets.IntInput(name='Random State', start=0, end=10, step=1)
        scoring= pn.widgets.Select(name='Select Scoring method', options=list(self.config_inquiry_engine['scoring'].keys()), value='neg_mean_absolute_percentage_error')
        self.optimization_additionals[tab_num].extend([n_jobs, n_splits, n_repeats, random_state, scoring])

        

        self.config_panel[tab_num] = pn.Column(component, max_width= 100)
        self.optimize_config_panel[tab_num] = pn.Column(component_optimized, self.optimization_additionals[tab_num], max_width= 100)

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

            if self.model_info[tab_num][hyperparameter] in [type(str), type(None)]:
                widget=  pn.widgets.TextInput(name=hyperparameter, placeholder='defaults to {}'.format(default_value))
                widget_box.append(widget)

            
            widget_optimizer = pn.widgets.TextInput(name=hyperparameter, placeholder= 'comma separated values or "-" separated range', value=None)
            widget_optimizer.param.watch(lambda event: self.optimizer_enabler(event, tab_num))
            
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
        self.optimize_config_panel[tab_num].append(self.optimize_evaluate_buttons[tab_num])


    def optimizer_enabler(self, tab_num):
        #Removing the evaluate button
        listt = self.optimize_config_panel[tab_num][0:-1]
        if (any(obj.value is not None for obj in listt)) & (tab_num in list(self.model_objects.keys())):
           self.optimize_evaluate_buttons[tab_num].disable= False
    
        
    def evaluate_normal(self, event, tab_num):
        hyperparameters = {}
        for hyperparameter in self.model_info[tab_num].keys():
            #searching for the widget in self.config_panel[tab_num] whose name is the same as the name of the hyperparameter
            value= self.config_panel[tab_num][self.config_panel[tab_num].name==hyperparameter].value
            hyperparameters[hyperparameter] = value

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


    def evaluate_optimized(self, event, tab_num):
        hyperparameters = {}
        for hyperparameter in self.model_info[tab_num].keys():
            #searching for the widget in self.config_panel[tab_num] whose name is the same as the name of the hyperparameter
            value= self.optimize_config_panel[self.optimize_config_panel.name==hyperparameter].value
            
            # Now splitting this value
            value= value.replace(" ", "")
            list1= value.split(',')
            hyperparameters[hyperparameter] = []
            for i in list1:
                list2= i.split('-')
                hyperparameters[hyperparameter].append(list2)
            
        # Specifying additional_models
        model_type= self.model_types[tab_num]
        new_data= self.data.filter(self.features_selector.value)
        target_column= self.target_column_selector.value
        train_size= self.train_size_selectors_optimization[tab_num][0]
        test_size= 1 - self.train_size_selectors_optimization[tab_num][0]
        random_state= self.random_state_selector_optimization[tab_num]

        self.optimized_model_objects[tab_num] = self.model_objects[tab_num].tune(data=new_data, target_column=target_column, model_type=model_type, random_state=random_state, **hyperparameters)

        train= self.optimized_model_objects[tab_num].data_for_graph['Train']
        validation= self.optimized_model_objects[tab_num].data_for_graph['Validation']
        test= self.optimized_model_objects[tab_num].data_for_graph['Test']
        
        self.plotly_optimization[tab_num]= self.make_plot_pane(tab_num, train, validation, test, tab_num)


    def make_plot_pane(self, tab_num:int, train:pd.DataFrame=None, validation:pd.DataFrame=None, test:pd.DataFrame=None):
        if not all([train, test, validation]):
            return pn.pane.Plotly('# We are still waiting for a response!')
        self.gridspecs[tab_num]= pn.GridSpec(mode='warn',height= 800)

        self.gridspecs[tab_num][0, 0:5] = self.plot_lineplot_only(df= train)
        self.gridspecs[tab_num][0, 5] = self.make_statsbox(train)

        self.gridspecs[tab_num][1, 0:5] = self.plot_lineplot_only(df= validation)
        self.gridspecs[tab_num][1, 5] = self.make_statsbox(validation)

        widget= pn.widgets.EditableFloatSlider(name='Enter Tolerence (%)', start=0, end=10, step=0.1, value=0)
        widget.param.watch(lambda event: self.change_lines(event, tab_num, test))

        self.gridspecs[tab_num][2, 0:5] = self.plot_lineplot(df=test, value= widget.value)
        self.gridspecs[tab_num][2, 5] = self.make_statsbox(test)
    
    def change_lines(self, event, tab_num):
        self.gridspecs[tab_num][2, 0:5] = self.plot_lineplot(self.optimized_model_objects[tab_num].data_for_graph['Test'], value= event.new)

        
    def plot_lineplot(self, df, value, event=None):

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
        fig.add_trace(go.Scatter(x=df.index, y=df['Actual'], name='Original_Visc', line=dict(color='blue')), secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df['Predictions'], name='new_preds', line=dict(color='red')), secondary_y=False)

        # Set titles
        fig.update_yaxes(title_text="<b>primary</b>", secondary_y=False)
        fig.update_yaxes(title_text="<b>secondary</b>", secondary_y=True)

        return fig
    
    def plot_lineplot_only(self, df, event=None):

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        fig.add_trace(go.Scatter(x=df.index, y=df['Actual'], name='Original_Visc', line=dict(color='blue')), secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df['Predictions'], name='new_preds', line=dict(color='red')), secondary_y=False)

        # Set titles
        fig.update_yaxes(title_text="<b>primary</b>", secondary_y=False)
        fig.update_yaxes(title_text="<b>secondary</b>", secondary_y=True)

        return fig

    def make_statsbox(self, data):
        score, mse, rmse, mape, mae = Regression.evaluate(data['Actual'],data['Predictions'])

        score_indicator = pn.indicators.Number(value=(score), default_color='black', name='R2 score', format='{value}')
        card_score= pn.Card(
                            score_indicator,
                            styles={'background': 'lightgray'},
                            hide_header=True
                            )
        
        mse_indicator = pn.indicators.Number(value=(mse), default_color='black', name='MSE', format='{value}')
        card_mse= pn.Card(
                            mse_indicator,
                            styles={'background': 'lightgray'},
                            hide_header=True
                            )
        
        rmse_indicator = pn.indicators.Number(value=(rmse), default_color='black', name='RMSE', format='{value}')
        card_rmse= pn.Card(
                            rmse_indicator,
                            styles={'background': 'lightgray'},
                            hide_header=True
                            )
        
        mape_indicator = pn.indicators.Number(value=(mape), default_color='black', name='MAPE', format='{value}')
        card_mape= pn.Card(
                            mape_indicator,
                            styles={'background': 'lightgray'},
                            hide_header=True
                            )
        
        mae_indicator = pn.indicators.Number(value=(mae), default_color='black', name='MAE', format='{value}')
        card_mae= pn.Card(
                            mae_indicator,
                            styles={'background': 'lightgray'},
                            hide_header=True
                            )
        
        return pn.Column(score_indicator, mse_indicator, rmse_indicator, mape_indicator, mae_indicator, scrollable=True)