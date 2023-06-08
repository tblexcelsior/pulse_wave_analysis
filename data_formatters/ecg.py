import data_formatters.base
import numpy as np
import pandas as pd
import libs.utils as utils
from sklearn.preprocessing import StandardScaler

InputTypes = data_formatters.base.InputType
GenericDataFormatter = data_formatters.base.GenericDataFormatter

class ECGFormatter(GenericDataFormatter):
    _column_definition = [
        ('p_id', InputTypes.ID),
        ('Time', InputTypes.TIME),
        ('ECG', InputTypes.TARGET)
    ]

    def __init__(self) -> None:
        """
            Initialize formatter
        """

        self.identifiers = None
        self.target_scalers = None
    
    def split_data(self, df, valid_boundary = 600, test_boundary = 100):
        idx = df['time_from_start']
        train_df = df[idx < valid_boundary]
        valid_df = df[(idx >= valid_boundary) & (idx < test_boundary)]
        test_df = df[idx > test_boundary]

        self.set_scalers(train_df)

        return (self.transform_input(data) for data in [train_df, valid_df, test_df])

    def set_scalers(self, df):
        
        columns_definitions = self.get_column_definition()
        id_column = utils.get_single_col_by_input_type(InputTypes.ID, columns_definitions)
        target_column = utils.get_single_col_by_input_type(InputTypes.TARGET, columns_definitions)

        self._target_scalers = {}
        identifiers = []

        for identifier, sliced in df.groupby(id_column):
            data = sliced[target_column].values.reshape((-1, 1))
            self._target_scalers[identifier] = StandardScaler().fit(data)
            identifiers.append(identifier)
        
        self.identifiers = identifiers

    def transform_input(self, df):
        
        if self._target_scalers is None:
            raise ValueError('Scaler has not been set')

        column_definition = self.get_column_definition()

        id_col = utils.get_single_col_by_input_type(InputTypes.ID, column_definition)
        target_column = utils.get_single_col_by_input_type(InputTypes.TARGET, column_definition)

        df_list = []

        for identifier, sliced in df.groupby(id_col):
            sliced_copy = sliced.copy()
            sliced_copy[target_column] = self._target_scalers[identifier].transform(sliced_copy[target_column].values.reshape((-1, 1)))
            df_list.append(sliced_copy)

        output = pd.concat(df_list, axis=0)

        return output

        
