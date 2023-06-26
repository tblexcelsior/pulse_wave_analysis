import data_formatters.base
import numpy as np
import pandas as pd
import libs.utils as utils
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder

InputTypes = data_formatters.base.InputType
GenericDataFormatter = data_formatters.base.GenericDataFormatter

class PiezoFormatter(GenericDataFormatter):
    _column_definition = [
        ('id', InputTypes.ID),
        ('time_from_start', InputTypes.TIME),
        ('ECG', InputTypes.TARGET)
    ]
    
    def __init__(self) -> None:
        """
            Initialize formatter
        """
        pass
    
    def split_data(self, df):
        train_df = []
        valid_df = []
        test_df = []
        for i, s in df.groupby('id'):
            idx = s['time_from_start']
            max_time = max(idx)
            train_portion = 0.8*max_time
            train_df.append(s[idx < train_portion])
            valid_portion = 0.9*max_time
            valid_df.append(s[((idx > train_portion) & (idx < valid_portion))])
            test_df.append(s[idx > valid_portion])
        train_df = pd.concat(train_df)
        valid_df = pd.concat(valid_df)
        test_df = pd.concat(test_df)

        self._set_scaler(train_df)

        return [self.transform_input(x) for x in [train_df, valid_df, test_df]]

    def _set_scaler(self, df):
        self._scaler = {}
        self._categorical_scaler = {}
        for (identifier, cate), sliced in df.groupby(['id', 'hour']):
            real_data = sliced['data'].values.reshape((-1, 1))
            self._scaler[(identifier, cate)] = StandardScaler().fit(real_data)

        self._categorical_scaler['hour'] = LabelEncoder().fit(df['hour'].apply(str).values.reshape((-1)))

    def transform_input(self, df):
        df_list = []
        for (identifier, cate), sliced in df.groupby(['id', 'hour']):
            sliced_copy = sliced.copy()
            sliced_copy['data'] = self._scaler[(identifier, cate)].transform(sliced_copy['data'].apply(str).values.reshape((-1, 1)))
            df_list.append(sliced_copy)

        output = pd.concat(df_list, axis=0)
        output['hour'] = self._categorical_scaler['hour'].transform(df['hour'].apply(str).values.reshape(-1))
        return output
    
    def data_preparation(self, df, max_samples, window_size, predicting_steps):
        sampling_localtions = []
        split_data_map = {}
        time_steps = predicting_steps + window_size

        for (identifier, cate), sliced in df.groupby(['id', 'hour']):
            num_entries = len(sliced)
            if num_entries >= time_steps:
                sampling_localtions += [(identifier, cate, time_steps + i) for i in range(num_entries - time_steps + 1)]
                split_data_map[(identifier, cate)] = sliced

        if max_samples > 0 and len(sampling_localtions) > max_samples:
            print('Extracting {} samples...'.format(max_samples))
            ranges = [
                sampling_localtions[i] for i in np.random.choice(
                    len(sampling_localtions), max_samples, replace=False)
            ]
        else:
            print('Max samples={} exceeds # available segments={}'.format(max_samples, len(sampling_localtions)))
            ranges = sampling_localtions
        inputs = np.zeros((max_samples, window_size, 1))
        known_inputs = np.zeros((max_samples, time_steps, 2))
        outputs = np.zeros((max_samples, predicting_steps, 1))
        for i, tup in enumerate(ranges):
            if (i + 1 % 1000) == 0:
                print(i + 1, 'of', max_samples, 'samples done...')
            identifier, cate, start_idx = tup
            sliced = split_data_map[(identifier, cate)].iloc[start_idx - time_steps:start_idx]
            inputs[i, :, :] = sliced[['data']][:window_size]
            known_inputs[i, :, :] = sliced[['time_from_start', 'hour']][:]
            outputs[i, :, :] = sliced[['data']][window_size:]
        return inputs, known_inputs, outputs

    def windowed_dataset(self, input_df, max_samples, window_size, prediction_steps, batch_size):
        inputs, known_inputs, outputs = self.data_preparation(input_df, max_samples, window_size, prediction_steps)
        inputs = tf.data.Dataset.from_tensor_slices(inputs)
        known_inputs = tf.data.Dataset.from_tensor_slices(known_inputs)
        outputs = tf.data.Dataset.from_tensor_slices(outputs)
        concat_inputs = tf.data.Dataset.zip((inputs, known_inputs))
        dataset = tf.data.Dataset.zip((concat_inputs, outputs))
        dataset = dataset.shuffle(max_samples)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

        
