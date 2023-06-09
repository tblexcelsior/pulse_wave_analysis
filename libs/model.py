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
  if dropout_rate is not None:
     inputs = tf.keras.layers.Dropout(dropout_rate)(inputs)
  x1 = tf.keras.layers.Dense(hidden_layer_size)(inputs)
  x2 = tf.keras.layers.Dense(hidden_layer_size, activation = 'sigmoid')(inputs)
  return tf.keras.layers.Multiply()([x2, x1])

def gated_residual_network(inputs, hidden_layer_size, dropout_rate, output_size = None, additional_inputs = None):
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
  hidden = gating_layer(hidden, output_size, dropout_rate)
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
    def __init__(self, raw_params, use_cudnn = False) -> None:

        params = dict(raw_params)
        self.time_steps = int(params['time_steps'])
        self.num_encoder_steps = int(params['num_encoder_steps'])
        self.wavelet_level = int(params['wavelet_level'])
        self.wavelet_type = params['wavelet_type']
        self.predicting_steps = int(params['predicting_steps'])
        self.hidden_layer_size = int(params['hidden_layer_size'])
        self.dropout_rate = float(params['dropout_rate'])
        self.use_cudnn = use_cudnn
        self.num_heads = int(params['num_heads'])
        self.output_size = int(params['output_size'])
        self.quantiles = [0.1, 0.5, 0.9]

        # self.max_gradient_norm = float(params['max_gradient_norm'])
        # self.learning_rate = float(params['learning_rate'])
        # self.minibatch_size = int(params['minibatch_size'])
        # self.num_epochs = int(params['num_epochs'])
        # self.early_stopping_patience = int(params['early_stopping_patience'])
        self.model = self.build_model()

    def _build_base_model(self):
        time_steps = self.time_steps
        wavelet_level = self.wavelet_level
        wavelet_type = self.wavelet_type

        inputs = Input(shape=(time_steps, 1))
        # Station wavelet transform
        swt_features = tf.keras.layers.Lambda(lambda x: swt_function(x, wavelet=wavelet_type, level=wavelet_level))(inputs)
        embedding_swt_features = tf.keras.backend.stack([
                                    convert_real_to_embedding(swt_features[Ellipsis, i : i + 1], self.hidden_layer_size) 
                                    for i in range(wavelet_level + 1)
                                    ], axis = -1)
        selected_wavelet_features = features_selection(embedding_swt_features, self.hidden_layer_size, self.dropout_rate)

        # Temporal processing
        historical_features = tf.keras.layers.Dense(self.hidden_layer_size)(inputs)
        
        def get_lstm():
          if self.use_cudnn:
            lstm = tf.compat.v1.keras.layers.CuDNNLSTM(
                self.hidden_layer_size,
                return_sequences=True,
                stateful=False,
            )
          else:
            lstm = tf.keras.layers.LSTM(
                self.hidden_layer_size,
                return_sequences=True,
                stateful=False,
                # Additional params to ensure LSTM matches CuDNN, See TF 2.0 :
                # (https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
                activation='tanh',
                recurrent_activation='sigmoid',
                recurrent_dropout=0,
                unroll=False,
                use_bias=True)
          return lstm
        historical_lstm = get_lstm()(historical_features)
        historical_lstm = gating_layer(historical_lstm, self.hidden_layer_size, self.dropout_rate)
        temporal_feature_layer = add_and_norm([historical_features, historical_lstm])
        
        wavelet_enhanced_layer = gated_residual_network(temporal_feature_layer, 
                                                        self.hidden_layer_size,
                                                        self.dropout_rate,
                                                        additional_inputs=selected_wavelet_features)
        x, attention_weights = tf.keras.layers.MultiHeadAttention(self.num_heads,
                                                                      self.hidden_layer_size // self.num_heads)(wavelet_enhanced_layer, 
                                                                                                                wavelet_enhanced_layer,
                                                                                                                wavelet_enhanced_layer,
                                                                                                                return_attention_scores = True)
        x = gating_layer(x,
                         self.hidden_layer_size,
                         dropout_rate=self.dropout_rate,
                         )
        x = add_and_norm([x, wavelet_enhanced_layer])

        decoder = gated_residual_network(x,
                                         self.hidden_layer_size,
                                         self.dropout_rate,
                                         )
        decoder = gating_layer(decoder, self.hidden_layer_size)

        transformer_layer = add_and_norm([decoder, temporal_feature_layer])
        model = tf.keras.Model(inputs, transformer_layer)
        return inputs, transformer_layer, attention_weights
        
    def build_model(self):
      inputs, transformer_layer, attention_weights = self._build_base_model()
      
      outputs = tf.keras.layers.Dense(self.output_size * len(self.quantiles)) \
                (transformer_layer[Ellipsis, self.num_encoder_steps:, :])
      
      model = Model(inputs=inputs, outputs=outputs)
      tf.keras.utils.plot_model(model, './model.png', show_shapes=True)
      print(model.summary())

    def input_pipeline(self, series):
        time_steps = self.time_steps
        predicting_steps = self.predicting_steps
        shuffle_buffer = len(series)
        minibatch_size = self.minibatch_size

        dataset = tf.data.Dataset.from_tensor_slices(series)
    
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