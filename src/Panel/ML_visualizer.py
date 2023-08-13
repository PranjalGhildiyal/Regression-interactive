import panel as pn
import pandas as pd
import re
from ..Exceptions import *


class Visualize_ML:
    def __init__(self, data:pd.DataFrame):

        self.data= Visualize_ML.cleaan_dataset(data)





    


    @staticmethod                   
    def cleaan_dataset(data:pd.DataFrame)->pd.DataFrame:
        all_columns = data.columns
        pattern = r'Unnamed: \d+|level_\d+|index'
        bad_columns = [col for col in all_columns if re.match(pattern, col)]

        return data.drop(bad_columns, axis = 1)
