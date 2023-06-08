import tensorflow as tf
from tensorflow.keras import Input, Model
import pywt

def swt_function(inputs, wavelet='sym5', level=5):
  def swt_wrapper(x):
    # Convert NumPy array to TensorFlow tensor
    x_tensor = tf.convert_to_tensor(x)

    # Perform SWT using pywt.swt
    swt_coeffs = pywt.swt(x_tensor.numpy(), wavelet=wavelet, level=level, trim_approx=True, norm=True)
    swt_concat = tf.concat([coeff for coeff in swt_coeffs], axis=-1)
    return swt_concat

  def wrapped_custom_function(inputs):
    outputs = tf.py_function(swt_wrapper, [inputs], tf.float32)
    return outputs

    # Convert TensorFlow tensor to NumPy array using tf.py_function
  input_shape = tf.shape(inputs)
  output_shape = (input_shape[0], 5120, level + 1)
  swt_output = tf.map_fn(wrapped_custom_function, inputs, dtype=tf.float32)
  swt_output = tf.reshape(swt_output, output_shape) 


  return swt_output

def add_and_norm(inputs_list):
  x = tf.keras.layers.Add()(inputs_list)
  x = tf.keras.layers.Normalization()(x)
  return x
def gating_layer(inputs, hidden_layer_size, dropout_rate = None):
  x1 = tf.keras.layers.Dense(hidden_layer_size)(inputs)
  x2 = tf.keras.layers.Dense(hidden_layer_size, activation = 'sigmoid')(inputs)
  return tf.keras.layers.Multiply()([x2, x1])

def gated_residual_network(inputs, hidden_layer_size, dropout_rate, output_size = None,additional_inputs = None):
  if output_size is None:
    output_size = hidden_layer_size
    skip = inputs
  else:
    linear = tf.keras.layers.Dense(output_size)
    skip = linear(inputs)

  hidden = tf.keras.layers.Dense(hidden_layer_size)(inputs)
  if additional_inputs is not None:
    hidden = hidden + tf.keras.layers.Dense(hidden_layer_size)(additional_inputs)
  hidden = tf.keras.layers.Activation('elu')(hidden)
  hidden = tf.keras.layers.Dense(hidden_layer_size)(hidden)
  hidden = tf.keras.layers.Dropout(dropout_rate)(hidden)
  hidden = gating_layer(inputs, output_size)
  hidden = add_and_norm([hidden, skip])
  return hidden

def features_selection(inputs, hidden_layer_size, dropout_rate):
  _, time_steps, num_dims, num_features = inputs.get_shape().as_list()
  flatten_inputs = tf.reshape(inputs, (-1, time_steps, num_features * num_dims))
  selection_weights = gated_residual_network(flatten_inputs, hidden_layer_size, dropout_rate, output_size = num_features)
  sparse_weights = tf.keras.layers.Activation('softmax')(selection_weights)
  sparse_weights = tf.expand_dims(sparse_weights, axis=2)
  transformed_x = []
  for i in range(num_features):
    e = gated_residual_network(inputs[Ellipsis, i], hidden_layer_size, dropout_rate)
    transformed_x.append(e)

  transformed_x = tf.keras.backend.stack(transformed_x, axis = -1)
  combined = tf.keras.layers.Multiply()([sparse_weights, transformed_x])
  selected_features = tf.keras.backend.sum(combined, axis = -1)
  return selected_features


def convert_real_to_embedding(inputs, hidden_layer_size):
    return tf.keras.layers.Dense(hidden_layer_size)(inputs)

class HopeThisWork(object):
    def __init__(self, raw_params) -> None:

        params = dict(raw_params)
        self.time_steps = int(params['time_steps'])
        self.wavelet_level = int(params['wavelet_level'])
        self.wavelet_type = params['wavelet_type']
        self.predicting_steps = int(params['predicting_steps'])
        self.hidden_layer_size = int(params['hidden_layer_size'])
        self.dropout_rate = float(params['dropout_rate'])
        
        # self.max_gradient_norm = float(params['max_gradient_norm'])
        # self.learning_rate = float(params['learning_rate'])
        # self.minibatch_size = int(params['minibatch_size'])
        # self.num_epochs = int(params['num_epochs'])
        # self.early_stopping_patience = int(params['early_stopping_patience'])

        self.model = self._build_base_model()

    def _build_base_model(self):
        time_steps = self.time_steps
        wavelet_level = self.wavelet_level
        wavelet_type = self.wavelet_type

        inputs = Input(shape=(time_steps, 1))
        swt_features = tf.keras.layers.Lambda(lambda x: swt_function(x, wavelet=wavelet_type, level=wavelet_level))(inputs)
        embedding_swt_features = tf.keras.backend.stack([
                                    convert_real_to_embedding(swt_features[Ellipsis, i : i + 1], self.hidden_layer_size) 
                                    for i in range(wavelet_level + 1)
                                    ], axis = -1)
        selected_wavelet_features = features_selection(embedding_swt_features, self.hidden_layer_size, self.dropout_rate)
        model = tf.keras.Model(inputs, selected_wavelet_features)
        tf.keras.utils.plot_model(model, './model1.png', show_shapes=True)
        
    def input_pipeline(self, series):
        time_steps = self.time_steps
        predicting_steps = self.predicting_steps
        shuffle_buffer = len(series)
        minibatch_size = self.minibatch_size

        datasets = tf.data.Dataset.from_tensor_slices(series)
    
        # Window the data but only take those with the specified size
        dataset = dataset.window(time_steps + predicting_steps, shift=1, drop_remainder=True)
        
        # Flatten the windows by putting its elements in a single batch
        dataset = dataset.flat_map(lambda window: window.batch(time_steps + predicting_steps))

        # Create tuples with features and labels 
        dataset = dataset.map(lambda window: (window[:time_steps], window[time_steps:predicting_steps]))

        # Shuffle the windows
        dataset = dataset.shuffle(shuffle_buffer)
        
        # Create batches of windows
        dataset = dataset.batch(minibatch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def frequency_selection(self, x):
        x = tf.signal.fft(x)