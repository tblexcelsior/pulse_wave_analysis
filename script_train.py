import pandas as pd
import data_formatters.piezo
import numpy as np
from libs.model import HopeThisWork

data_csv_path = "datasets/piezo/over_extended_piezo.csv"
INPUT_LENGTH = 128*6
OUTPUT_LENGTH = 128
BATCH_SIZE = 128
TRAIN_SAMPLES = 400000
VALID_SAMPLES = 50000
raw_params = {
    'time_steps': INPUT_LENGTH + OUTPUT_LENGTH,
    'output_length': OUTPUT_LENGTH,
    'hidden_layer_size': 160,
    'dropout_rate': 0.1,
    'num_heads': 5,
    'max_gradient_norm': 0.01,
    'learning_rate': 0.0005,
    'num_epochs': 50,
    'early_stopping_patience': 5,
    'minibatch_size': BATCH_SIZE,
    'model_folder': 'outputs'
              }
raw_data = pd.read_csv(data_csv_path)
data_formatter = data_formatters.piezo.PiezoFormatter()
ModelClass = HopeThisWork(raw_params, True)
train_df, valid_df, _ = data_formatter.split_data(raw_data, 400, 500)

for _ in range(50):
    train_dataset = data_formatter.windowed_dataset(train_df, TRAIN_SAMPLES, INPUT_LENGTH, OUTPUT_LENGTH, BATCH_SIZE)
    valid_dataset = data_formatter.windowed_dataset(valid_df, VALID_SAMPLES, INPUT_LENGTH, OUTPUT_LENGTH, BATCH_SIZE)
    ModelClass.fit(train_dataset, valid_dataset)

ModelClass.save('outputs')