'''
---------------------------------
Developed by: Pranjal Ghildiyal
Date modified: 8/10/2023
---------------------------------
'''



import pandas as pd
import os
import panel as pn
import yaml
from matplotlib.cbook import boxplot_stats




class Analyze():
    def __init__(self, df:pd.DataFrame)->None:
        self.dataset = df
        self.clean_dataset()
        self.properties=list(self.dataset.columns)
        global current_dir
        current_dir= os.path.dirname(os.path.abspath(__file__))
        
        # Taking user input for Datetime column
        column_verification = False
        while not column_verification:
            self.datetime_column = input('Datetime column(press enter if not present):')
            if self.datetime_column in (list(self.dataset.columns) + ['']):
                column_verification = True

        if self.datetime_column != '':
            self.dataset[self.datetime_column] = pd.to_datetime(self.dataset[self.datetime_column])
            max_datetime = self.dataset[self.datetime_column].max()
            min_datetime = self.dataset[self.datetime_column].min()
            self.dataset = self.dataset.set_index(self.datetime_column)
            self.properties.remove(self.datetime_column)
            
            self.datetime_picker = pn.widgets.DatetimeRangePicker(value=(min_datetime, min_datetime),
                                            start=min_datetime.date(), end=max_datetime.date(),
                                            military_time=False, name='Datetime Range')
            self.datetime_picker.param.watch(self.__update_timestamps, 'value')

        # The date time column has been verified and the datetime widget has been built.

    

        


    def create_dashboard(self)->None:
        
        logo = pn.pane.PNG(object=os.path.join(current_dir, 'algo8_logo.png'), width = 300, height = 300)
        self.logo_pane = pn.Column(
                                    pn.pane.HTML('', style={'margin-top': '5px'}),  # Top padding
                                    logo,
                                    pn.pane.HTML('', style={'margin-bottom': '5px'})  # Bottom padding,
                                    
                                    )      

        # Defining all the major buttons here as dicts
        self.statistic_prop_selectors = {}
        self.undo_buttons = {}
        self.undo_all_buttons = {}
        self.redo_all_buttons= {}
        self.redo_buttons = {}
        self.evaluate_buttons={}
        self.outlier_buttons = {}
        self.sliders = {}
        self.nan_buttons = {}
        self.keys = {}
        self.operation_buttons = {}
        self.next_tab_num=1
        self.value_inputs = {}
        self.current_dataset = self.dataset.copy(deep=True)
        self.property_tabs = {}
        self.scrollable_buttons = {}
        self.buttons_exist = []
        self.scrollable_elements = {}
        self.last_button = 0
        

        # Making buttons for statistical tab (one time only)
        self.undo_buttons[0] = pn.widgets.Button(name='Undo', button_type='primary')
        self.redo_buttons[0] = pn.widgets.Button(name='Redo', button_type='primary')
        self.nan_buttons[0] = pn.widgets.Button(name='Remove NaN', button_type='primary')
        self.evaluate_buttons[0] = pn.widgets.Button(name = 'Evaluate', button_type='success')
        self.redo_all_buttons[0] = pn.widgets.Button(name='Redo All', button_type='danger')
        self.undo_all_buttons[0] = pn.widgets.Button(name='Undo All', button_type='danger')
        self.evaluate_all = pn.widgets.Button(name='Evaluate All', button_type='primary')

        # Adding watcher functions here
        self.undo_buttons[0].on_click(self.undo_for_statistical)
        self.redo_buttons[0].on_click(self.redo_for_statistical)
        self.nan_buttons[0].on_click(self.remove_all_nan)
        self.evaluate_buttons[0].on_click(self.evaluate_func)
        self.redo_all_buttons[0].on_click(self.redo_all)
        self.undo_all_buttons[0].on_click(self.undo_all)
        self.evaluate_all.on_click(self.evaluate_all_func)
        self.scrollable_elements[0] = pn.Column(sizing_mode='scale_width', max_height=300)
        

        

        self.statistic_prop_selectors[0] = pn.widgets.Select(name='', options=['None'] + self.properties, value=None)
        self.statistic_prop_selectors[1] = pn.widgets.Select(name='', options=['None'] + self.properties, value=None)
        self.property_selector = pn.widgets.Select(name='', options=['None'] + self.properties, value=None)

        # Adding watcher functions to above:
        self.property_selector.param.watch(self.update_property, 'value')
        self.statistic_prop_selectors[0].param.watch(self.__change_statistics_property, 'value')
        self.statistic_prop_selectors[1].param.watch(self.__change_statistics_property, 'value')

        # Defining a set of operations.
        self.operations= pd.DataFrame(columns=['property', 'operation', 'args', 'display_name', 'switch'])

        self.statistics_tab = self.__make_statistics_tab()
        
        self.tabs = pn.Tabs(('Comparision', self.statistics_tab),
                        closable = False)
        

        self.color_coding = {
                        'self.range_cleaning': ['default','success'],
                    'self.remove_nans': ['default','success'],
                    'self.remove_values': ['default','success'],
                    'self.value_cleaning': ['default','success'],
                    'self.drop_all_nans': ['default', 'success']
                    }

        if self.datetime_column != '':
            self.sidebar = pn.Column(
                                pn.Row(self.logo_pane),
                                pn.Row(self.datetime_picker),
                                pn.Row(self.property_selector),
                                self.scrollable_elements[0]
                                )
        else:
            self.sidebar = pn.Column(
                                pn.Row(self.logo_pane),
                                pn.Row(self.property_selector),
                                self.scrollable_elements[0]
                                )
        
        self.template = pn.template.BootstrapTemplate(
                                                    title="Plant Brain",
                                                    sidebar=self.sidebar,
                                                    header_color = '#FFFFFF',
                                                    background_color = '#FFFFFF',
                                                    corner_radius = 1,
                                                    main_layout = ''
                                                )
        self.template.header_background = '#0a76b7'
        
        self.template.main.append(self.tabs)
        
        pn.serve(self.template, show=True)

    def __change_statistics_property(self, event=None):

        self.statistics_tab.object = self.__make_statistics_tab()


    def __update_timestamps(self, event):
        start = event.new[0]
        end = event.new[1]
        operation_series = pd.Series([self.datetime_column, self.range_cleaning, [self.datetime_column] + [start, end], 'Filter {}'.format(self.datetime_column), 1], index = ['property', 'operation', 'args', 'display_name', 'switch'])
        self.operations= self.operations.append(operation_series, ignore_index = True)
        max_index = max(self.operations.index)
        self.scrollable_buttons[max_index] = pn.widgets.Button(name='Filter {}'.format(self.datetime_column), button_type='success')
        self.scrollable_buttons[max_index].on_click(lambda event1: self.scrollable_buttons_watcher(event1, max_index))
        self.__change_markdown()

        

    def evaluate_func(self, event):
        self.reperform_all_operations()
        self.tabs[0].object = self.__make_statistics_tab()
    
    def evaluate_all_func(self, event):
        self.reperform_all_operations()
        self.tabs[0].object= self.__make_statistics_tab()
        for key in self.buttons_exist:
            self.tabs[key].object = self.__make_property_tab(self.keys[key])
        
        self.tabs.active=0

    def undo_all(self, event):
        self.operations['switch'] = 0
        for button in self.scrollable_buttons.keys():
            self.scrollable_buttons[button].button_type= 'default'

    def redo_all(self, event):
        self.operations['switch'] = 1
        for button in self.scrollable_buttons.keys():
            self.scrollable_buttons[button].button_type= self.color_coding['self.' + self.operations.loc[button, 'operation'].__func__.__name__][1]

    def undo_for_statistical(self, event):

        if self.operations[(self.operations.switch == 1)].shape[0]> 0:
            max_index= max(self.operations[(self.operations.switch == 1)].index)
            self.operations.loc[max_index, 'switch'] = 0
            self.scrollable_buttons[max_index].button_type= 'default'
    
    def redo_for_statistical(self, event):

        if self.operations[(self.operations.switch == 0)].shape[0]> 0:
            max_index= max(self.operations[(self.operations.switch == 0)].index)
            self.operations.loc[max_index, 'switch'] = 1
            self.scrollable_buttons[max_index].button_type= self.color_coding['self.' + self.operations.loc[max_index, 'operation'].__func__.__name__][1]

    


        


    def remove_all_nan(self, event):
        new_row=pd.Series(['', self.drop_all_nans, [], 'Remove all NaN', 1], index = ['property', 'operation', 'args', 'display_name', 'switch'])
        self.operations = self.operations.append(new_row, ignore_index = True)
        max_index= max(self.operations.index)

        self.scrollable_buttons[max_index] = pn.widgets.Button(name='Removed All NaN', button_type='success')
        self.scrollable_buttons[max_index].on_click(lambda event1: self.scrollable_buttons_watcher(event1, max_index))
        self.__change_markdown()


    def __make_statistics_tab(self):
        prop1 = self.statistic_prop_selectors[0].value
        prop2 = self.statistic_prop_selectors[1].value
        
        gspec = pn.GridSpec(mode='warn', name = 'Comparisions')
        
        if (prop1 is None) and (prop2 is None):
            return gspec
        
        config_tab = pn.Column(
                                pn.Row(self.statistic_prop_selectors[0], pn.pane.Markdown('vs'), self.statistic_prop_selectors[1], sizing_mode = 'scale_width'),
                                self.nan_buttons[0],
                                pn.Row(self.undo_buttons[0], self.redo_buttons[0], sizing_mode = 'scale_width'),
                                pn.Row(self.undo_all_buttons[0], self.redo_all_buttons[0], sizing_mode = 'scale_width'),
                                pn.Row(self.evaluate_buttons[0], self.evaluate_all, sizing_mode = 'scale_width'),
                                pn.Column('Operations: ', self.scrollable_elements[0],sizing_mode='scale_width', max_height=300),
                                sizing_mode='scale_width'
                                        
        )
        gspec[0, 1] = config_tab

        

        # Now making plots #####
        

        # Make the required grid here
        return gspec


    def update_property(self, event, property=None):

        
        # Chech whether this property tab is already available:
        if property is None:
            value= event.new
        else:
            value=property

        # Commit all the changes
        self.reperform_all_operations()
        
        if value is not None:
            property_available= value in list(self.keys.keys())
            
            if property_available:
                # Making the tab with property active if the property is already present
                tab_num = self.keys[value]
                self.tabs[tab_num].object = self.__make_property_tab(value)
                self.tabs.active = tab_num
            
            else:
                key = self.next_tab_num
                self.next_tab_num += 1

                self.keys[key] = event.new
                self.keys[value] = key

                self.make_property_buttons(key)
                new_tab = self.__make_property_tab(value)
                self.tabs.append(new_tab)
                self.tabs.active= key
                
    def make_property_buttons(self, key):

        property = self.keys[key]

        # Here, make the property tab
        
        if (self.dataset[property].dtype.name == 'category'):
            # Adding relevant buttons for categorical type:
            self.sliders[key] = pn.widgets.DiscreteSlider(name='Select Category', options= ['ALL'] + list(self.dataset[property].unique()), value='ALL')
            self.value_inputs[key] = pn.widgets.AutocompleteInput( name='', options=['ALL'] + list(self.dataset[property].astype(str).unique()), placeholder='Select Category', value='ALL')
            
            # Adding watchers for them
            self.sliders[key].param.watch(lambda event1: self.categorical_slider_watcher(event1, property), 'value')
            self.value_inputs[key].param.watch(lambda event1: self.watch_categorical_inputs(event1, property), 'value')
        else:
            self.sliders[key] = pn.widgets.RangeSlider(value=(self.dataset[property].min(), self.dataset[property].max()), start=self.dataset[property].min(), end=self.dataset[property].max(), step=((self.dataset[property].max() - self.dataset[property].min()) / 10), name="Select Range")
            self.value_inputs[key] = pn.widgets.TextInput(placeholder='Enter range of values (separated by "-")')

            # Adding watchers for them
            self.sliders[key].param.watch(lambda event1: self.range_slider_watcher(event1, property), 'value')
            self.value_inputs[key].param.watch(lambda event1: self.watch_limits(event1, property), 'value')

        self.evaluate_buttons[key] = pn.widgets.Button(name='Evaluate', button_type='success')
        self.nan_buttons[key] = pn.widgets.Button(name='Remove NaN', button_type='primary')
        self.outlier_buttons[key] = pn.widgets.Button(name='Remove Outliers', button_type= 'primary')

        self.evaluate_buttons[key].on_click(lambda event1: self.update_property(event1, property=property))
        self.nan_buttons[key].on_click(lambda event1: self.remove_nan(event1, property))

        # Making redo and undo buttons
        self.undo_buttons[key] = pn.widgets.Button(name= 'Undo', button_type='primary')
        self.redo_buttons[key] = pn.widgets.Button(name='Redo', button_type= 'primary')
        self.undo_all_buttons[key] = pn.widgets.Button(name= 'Undo All', button_type='primary')
        self.redo_all_buttons[key] = pn.widgets.Button(name='Redo All', button_type= 'primary')
        self.outlier_buttons[key].on_click(lambda event: self.__remove_outliers(event, property))
        
        self.undo_buttons[key].on_click(lambda event1: self.undo_for_property(event1, property))
        self.redo_buttons[key].on_click(lambda event1: self.redo_for_property(event1, property))

        self.undo_all_buttons[key].on_click(lambda event1: self.undo_all_for_property(event1, property))
        self.redo_all_buttons[key].on_click(lambda event1: self.redo_all_for_property(event1, property))

        self.scrollable_elements[key] = pn.Column(sizing_mode='scale_width', max_height=300)


        # Registering the new buttons
        self.buttons_exist.append(key)

    
    
    def __make_property_tab(self, property):


        key= self.keys[property]
        # Now making a new tab
        gspec = pn.GridSpec(mode='warn', name=property)

        config_tab = pn.Column(
                                self.sliders[key],
                                self.value_inputs[key],
                                pn.Row(self.nan_buttons[key], self.outlier_buttons[key], sizing_mode = 'scale_width'),
                                pn.Row(self.undo_buttons[key], self.redo_buttons[key], sizing_mode = 'scale_width'),
                                pn.Row(self.undo_all_buttons[key], self.redo_all_buttons[key], sizing_mode = 'scale_width'),
                                self.evaluate_buttons[key],
                                pn.Column('Operations: ', self.scrollable_elements[key],sizing_mode='scale_width', max_height=300)
                                        
        )
        gspec[0, 1] = config_tab


        # Now making plots ####
        

        return gspec


    def undo_for_property(self, event, property):

        if self.operations[(self.operations.property == property) & (self.operations.switch == 1)].shape[0]> 0:
            max_index= max(self.operations[(self.operations.property == property) & (self.operations.switch == 1)].index)
            self.operations.loc[max_index, 'switch'] = 0
            self.scrollable_buttons[max_index].button_type='default'
    
    def redo_for_property(self, event, property):

        if self.operations[(self.operations.property == property) & (self.operations.switch == 0)].shape[0]> 0:
            max_index= max(self.operations[(self.operations.property == property) & (self.operations.switch == 0)].index)
            self.operations.loc[max_index, 'switch'] = 1
            self.scrollable_buttons[max_index].button_type='success'

    def undo_all_for_property(self, event, property):
        if self.operations[self.operations.property == property].shape[0] > 0:
            index=self.operations[self.operations.property == property].index
            self.operations.loc[self.operations.property == property, 'switch'] = 0
            for button in  index:
                self.scrollable_buttons[button].button_type = 'default'

    def redo_all_for_property(self, event, property):
        if self.operations[self.operations.property == property].shape[0] > 0:
            index = self.operations[self.operations.property == property].index
            self.operations.loc[self.operations.property == property, 'switch'] = 1
            for button in  index:
                self.scrollable_buttons[button].button_type = 'success'
        
    
    def __remove_outliers(self, event, property):
        l = list(boxplot_stats(self.current_dataset[property]).pop(0)['fliers'])
        new_row = pd.Series([property, self.remove_values, [property, l], 'Removed Outliers for {}'.format(property), 1], index = ['property', 'operation', 'args', 'display_name', 'switch'])
        self.operations = self.operations.append(new_row, ignore_index= True)

        max_index = max(self.operations.index)
        self.scrollable_buttons[max_index] = pn.widgets.Button(name='Removed Outliers for {}'.format(property), button_type='success')
        self.scrollable_buttons[max_index].on_click(lambda event1: self.scrollable_buttons_watcher(event1, max_index))
        self.__change_markdown()
            
    def remove_nans(self, column):
        if self.datetime_column != '':
            self.current_dataset = self.current_dataset.reset_index()
        self.current_dataset = self.current_dataset.dropna(subset = column)
        if self.datetime_column != '':
            self.current_dataset = self.current_dataset.set_index(self.datetime_column)

    def drop_all_nans(self):
        self.current_dataset = self.current_dataset.dropna()    
        
    def remove_values(self, column, value_list):
        if self.datetime_column != '':
            self.current_dataset = self.current_dataset.reset_index()
        self.current_dataset = self.current_dataset[~self.current_dataset[column].isin(value_list)]
        if self.datetime_column != '':
            self.current_dataset = self.current_dataset.set_index(self.datetime_column)
        
    def range_cleaning(self, column, start, end):
        if self.datetime_column != '':
            self.current_dataset = self.current_dataset.reset_index()
        self.current_dataset = self.current_dataset[self.current_dataset[column].between(start, end)]
        if self.datetime_column != '':
            self.current_dataset = self.current_dataset.set_index(self.datetime_column)
        
    def value_cleaning(self, column, value):
        if self.datetime_column != '':
            self.current_dataset = self.current_dataset.reset_index()
        self.current_dataset = self.current_dataset[self.current_dataset[column]==value]
        if self.datetime_column != '':
            self.current_dataset = self.current_dataset.set_index(self.datetime_column)




    def reperform_all_operations(self):
        '''Performs all the operations where switch==1'''
        self.current_dataset = self.dataset.copy(deep=True)
        for i in list(self.operations[self.operations.switch==1].index):
            self.scrollable_buttons[i].button_type= 'primary'
            self.operations.loc[i, 'operation'](*self.operations.loc[i, 'args'])
            

    def categorical_slider_watcher(self, event, property):
        if event.new != 'ALL':
            # Here, we will update all the data
            new_row=pd.Series([property, self.value_cleaning, [property] + [event.new], 'Filter {}'.format(property), 1], index = ['property', 'operation', 'args', 'display_name', 'switch'])
            self.operations = self.operations.append(new_row, ignore_index = True)
            # Making a new button corresponding to the new operation added
            max_index = max(self.operations.index)
            self.scrollable_buttons[max_index] = pn.widgets.Button(name='Filter {}'.format(property).format(property), button_type='success')
            self.scrollable_buttons[max_index].on_click(lambda event1: self.scrollable_buttons_watcher(event1, max_index))
            self.__change_markdown()
        else:
            if self.operations[(self.operations.property == property) & (self.operations.operation == self.value_cleaning)].shape[0] > 0:
                index=self.operations[(self.operations.property == property) & (self.operations.operation == self.value_cleaning)].index

                # Switching any prior operations off
                self.operations.loc[(self.operations.property == property) & (self.operations.operation == self.value_cleaning), 'switch'] = 0

                # Making the buttons fade
                for i in index:
                    self.scrollable_buttons[i].button_type='default'

    def watch_categorical_inputs(self, event, property):
        if event.new != 'ALL':
            # Here, we will update all the data
            new_row=pd.Series([property, self.value_cleaning, [property] + [event.new], 'Filter {}'.format(property), 1], index = ['property', 'operation', 'args', 'display_name', 'switch'])
            self.operations = self.operations.append(new_row, ignore_index = True)
            # Making a new button
            max_index = max(self.operations.index)
            self.scrollable_buttons[max_index] = pn.widgets.Button(name='Filter {}'.format(property), button_type='success')
            self.scrollable_buttons[max_index].on_click(lambda event1: self.scrollable_buttons_watcher(event1, max_index))
            self.__change_markdown()
        else:
            if self.operations[(self.operations.property == property) & (self.operations.operation == self.value_cleaning)].shape[0] > 0:
                index=self.operations[(self.operations.property == property) & (self.operations.operation == self.value_cleaning)].index
                self.operations.loc[(self.operations.property == property) & (self.operations.operation == self.value_cleaning), 'switch'] = 0

                for i in index:
                    self.scrollable_buttons[i].button_type='default'
    
    def range_slider_watcher(self, event, property):
        start, end = event.new
        event = (start, end)
        if event != (self.dataset[property].min(), self.dataset[property].max()):
            new_row=pd.Series([property, self.range_cleaning, [property] + [event[0], event[1]], 'Filter {}'.format(property), 1], index = ['property', 'operation', 'args', 'display_name', 'switch'])
            self.operations = self.operations.append(new_row, ignore_index=True)

            max_index= max(self.operations.index)
            self.scrollable_buttons[max_index] = pn.widgets.Button(name='Filter {}'.format(property), button_type='success')
            self.scrollable_buttons[max_index].on_click(lambda event1: self.scrollable_buttons_watcher(event1, max_index))
            self.__change_markdown()
        else:
            if self.operations[(self.operations.property == property) & (self.operations.operation == self.range_cleaning)].shape[0] > 0:
                index= self.operations[(self.operations.property == property) & (self.operations.operation == self.range_cleaning)].index
                self.operations.loc[(self.operations.property == property) & (self.operations.operation == self.range_cleaning), 'switch'] = 0

                for i in index:
                    self.scrollable_buttons[i].button_type='default'

    def watch_limits(self, event, property):

        try:
            a = event.new.split('-')[0]
            a = a.strip()
            if a == '':
                a = self.current_dataset[property].min()
            a = float(a)

            b = event.new.split('-')[1]
            b = b.strip()
            if b == '':
                b = self.current_dataset[property].min()
            b = float(b)

            event= (a, b)
            if event != (self.dataset[property].min(), self.dataset[property].max()):
                new_row=pd.Series([property, self.range_cleaning, [property] + [event[0], event[1]], 'Filter {}'.format(property), 1], index = ['property', 'operation', 'args', 'display_name', 'switch'])
                self.operations = self.operations.append(new_row, ignore_index=True)

                max_index = max(self.operations.index)
                self.scrollable_buttons[max_index] = pn.widgets.Button(name= 'Filter {}'.format(property), button_type='success')
                self.scrollable_buttons[max_index].on_click(lambda event1: self.scrollable_buttons_watcher(event1, max_index))
                self.__change_markdown()

            else:
                if self.operations[(self.operations.property == property) & (self.operations.operation == self.range_cleaning)].shape[0] > 0:
                    index = self.operations[(self.operations.property == property) & (self.operations.operation == self.range_cleaning)].index
                    self.operations.loc[(self.operations.property == property) & (self.operations.operation == self.range_cleaning), 'switch'] = 0

                    for i in index:
                        self.scrollable_buttons[i].button_type='default'
        
        except ValueError:
            key=self.keys[property]
            self.value_inputs[key].value = ''
            self.value_inputs[key].placeholder='Wrong Format Entered. Enter range of values (separated by "-")'


    def remove_nan(self, event, property):
        new_row=pd.Series([property, self.remove_nans, [property] , 'Remove NaN values {}'.format(property), 1], index = ['property', 'operation', 'args', 'display_name', 'switch'])
        self.operations = self.operations.append(new_row, ignore_index=True)

        max_index = max(self.operations.index)
        self.scrollable_buttons[max_index] = pn.widgets.Button(name='Remove NaN values {}'.format(property), button_type='success')
        self.scrollable_buttons[max_index].on_click(lambda event1: self.scrollable_buttons_watcher(event1, max_index))
        self.__change_markdown()




    def __change_markdown(self):

        for i in range(self.last_button, max(self.operations.index)+1, 1):
            self.scrollable_elements[0].append(self.scrollable_buttons[i])
            property = self.operations.loc[i, 'property']
            if property in self.keys.keys():
                key= self.keys[property]
                self.scrollable_elements[key].append(self.scrollable_buttons[i])
        self.last_button= max(self.operations.index) + 1


            

    def clean_dataset(self):
        all_columns = self.dataset.columns
        pattern = r'Unnamed: \d+|level_\d+|index'
        bad_columns = [col for col in all_columns if re.match(pattern, col)]
        self.dataset = self.dataset.drop(bad_columns, axis = 1)

    def scrollable_buttons_watcher(self, event, index):
        if self.scrollable_buttons[index].button_type!='default':
            self.scrollable_buttons[index].button_type='default'
            self.operations.loc[index, 'switch'] = 0
        else:
            color = self.color_coding['self.' + self.operations.loc[index, 'operation'].__func__.__name__][1]
            self.scrollable_buttons[index].button_type=color
            self.operations.loc[index, 'switch'] = 1

    def get_operations(self):
        if self.operations.shape[0] > 0:
            out=self.operations.copy(deep=True)
            out['operation_names'] = out.operation.apply(lambda x: x.__name__)
            out = out[['operation_names', 'args']]
            data={}
            for index, row in out.iterrows():
                data[index] = {'operation_names': row['operation_names'], 'args': row['args']}
            return yaml.dump(data)
        else:
            return None
        
    def get_operations_dict(self):
        if self.operations.shape[0] > 0:
            out=self.operations.copy(deep=True)
            out['operation_names'] = out.operation.apply(lambda x: x.__name__)
            out = out[['operation_names', 'args']]
            data={}
            for index, row in out.iterrows():
                data[index] = {'operation_names': row['operation_names'], 'args': row['args']}
            return data
        else:
            return None