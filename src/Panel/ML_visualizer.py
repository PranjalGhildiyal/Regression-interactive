import panel as pn
import pandas as pd
import re
from ..Exceptions import *


class Visualize_ML:
    def __init__(self, data:pd.DataFrame):

        self.data= Visualize_ML.cleaan_dataset(data)

        # Taking user input for Datetime column
        self.column_verification = False
        while not self.column_verification:
            self.datetime_column = input('Datetime column(press enter if not present):')
            if self.datetime_column in (list(self.dataset.columns) + ['']):
                self.column_verification = True

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




    


    @staticmethod                   
    def cleaan_dataset(data:pd.DataFrame)->pd.DataFrame:
        all_columns = data.columns
        pattern = r'Unnamed: \d+|level_\d+|index'
        bad_columns = [col for col in all_columns if re.match(pattern, col)]

        return data.drop(bad_columns, axis = 1)
