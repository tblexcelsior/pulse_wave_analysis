import shutil
import os
import tensorflow as tf
import libs.utils as utils
import pandas as pd
from tensorflow.keras import Input, Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Dense, Add, Normalization, Dropout, Multiply, TimeDistributed, Activation, LSTM

def add_and_norm(inputs_list):
  x = Add()(inputs_list)
  x = Normalization()(x)
  return x
def gating_layer(inputs, hidden_layer_size, dropout_rate = None):
  if dropout_rate is not None:
     inputs = tf.keras.layers.Dropout(dropout_rate)(inputs)
  x1 = Dense(hidden_layer_size)(inputs)
  x2 = Dense(hidden_layer_size, activation = 'sigmoid')(inputs)
  return Multiply()([x2, x1])

# Attention Components.
def get_decoder_mask(self_attn_inputs):
  """Returns causal mask to apply for self-attention layer.

  Args:
    self_attn_inputs: Inputs to self attention layer to determine mask shape
  """
  len_s = tf.shape(input=self_attn_inputs)[1]
  bs = tf.shape(input=self_attn_inputs)[:1]
  mask = tf.math.cumsum(tf.eye(len_s, batch_shape=bs), 1)
  return mask

def gated_residual_network(inputs, hidden_layer_size, dropout_rate, output_size = None, additional_inputs = None):
  if output_size is None:
    output_size = hidden_layer_size
    skip = inputs
  else:
    linear = Dense(output_size)
    skip = linear(inputs)

  hidden = Dense(hidden_layer_size)(inputs)
  if additional_inputs is not None:
    hidden = hidden + tf.keras.layers.Dense(hidden_layer_size)(additional_inputs)
  hidden = Activation('elu')(hidden)
  hidden = Dense(hidden_layer_size)(hidden)
  hidden = Dropout(dropout_rate)(hidden)
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

class T2V(tf.keras.layers.Layer):
    
    def __init__(self, output_dim=None, **kwargs):
        self.output_dim = output_dim
        super(T2V, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name='W',
                      shape=(input_shape[-1], self.output_dim),
                      initializer='uniform',
                      trainable=True)
        self.P = self.add_weight(name='P',
                      shape=(input_shape[1], self.output_dim),
                      initializer='uniform',
                      trainable=True)
        self.w = self.add_weight(name='w',
                      shape=(input_shape[1], 1),
                      initializer='uniform',
                      trainable=True)
        self.p = self.add_weight(name='p',
                      shape=(input_shape[1], 1),
                      initializer='uniform',
                      trainable=True)
        super(T2V, self).build(input_shape)
        
    def call(self, x):
        
        original = self.w * x + self.p
        sin_trans = K.sin(K.dot(x, self.W) + self.P)
        
        return K.concatenate([sin_trans, original], -1)

def convert_real_to_embedding(inputs, hidden_layer_size):
    return tf.keras.layers.Dense(hidden_layer_size)(inputs)

class HopeThisWork(object):
    def __init__(self, raw_params, use_cudnn = False) -> None:
        self.name = self.__class__.__name__
        params = dict(raw_params)
        self.time_steps = int(params['time_steps'])
        self.output_length = int(params['output_length'])
        self.hidden_layer_size = int(params['hidden_layer_size'])
        self.dropout_rate = float(params['dropout_rate'])
        self.use_cudnn = use_cudnn
        self.num_heads = int(params['num_heads'])
        self.minibatch_size = int(params['minibatch_size'])

        self.max_gradient_norm = float(params['max_gradient_norm'])
        self.learning_rate = float(params['learning_rate'])
        self.num_epochs = int(params['num_epochs'])
        self.early_stopping_patience = int(params['early_stopping_patience'])

        self.category_counts = 4

        self._temp_folder = os.path.join(params['model_folder'], 'tmp')
        self._score_folder = os.path.join(params['model_folder'], 'results')
        self.reset_temp_folder()

        self.model = self.build_model()

    def _build_base_model(self):
        time_steps = self.time_steps
        output_length = self.output_length

        real_inputs = Input(shape=(time_steps - output_length, 1))
        known_inputs = Input(shape=(time_steps, 2))

        time_inputs = tf.expand_dims(known_inputs[Ellipsis, :, 0], axis = -1)
        categorical_inputs = tf.expand_dims(known_inputs[Ellipsis, :, 1], axis = -1)
        # Temporal processing
        real_features = TimeDistributed(Dense(self.hidden_layer_size))(real_inputs)
        real_features = tf.expand_dims(real_features, axis = -1)

        time_features = T2V(self.hidden_layer_size - 1)(time_inputs)
        historical_time = tf.expand_dims(time_features[Ellipsis, : time_steps - output_length, :], axis = -1)
        future_time = tf.expand_dims(time_features[Ellipsis, time_steps - output_length : , :], axis = -1)

        embedding = tf.keras.Sequential([
                  tf.keras.layers.InputLayer([self.time_steps]),
                  tf.keras.layers.Embedding(
                      self.category_counts,
                      self.hidden_layer_size,
                      input_length=self.time_steps,
                      dtype=tf.float32)
                    ])(categorical_inputs)
        
        historical_embedding = tf.expand_dims(embedding[Ellipsis, : time_steps - output_length, :], axis = -1)
        future_embedding = tf.expand_dims(time_features[Ellipsis, time_steps - output_length : , :], axis = -1)

        historical_features = K.concatenate([real_features, historical_time, historical_embedding], -1)
        historical_features = features_selection(historical_features, self.hidden_layer_size, self.dropout_rate)

        future_features = K.concatenate([future_time, future_embedding], axis = -1)
        future_features = features_selection(future_time, self.hidden_layer_size, self.dropout_rate)
        skip = tf.concat([historical_features, future_features], axis = 1)

        def get_lstm(return_sequences = False, return_state = False):
          if self.use_cudnn:
            lstm = tf.compat.v1.keras.layers.CuDNNLSTM(
                self.hidden_layer_size,
                return_sequences=return_sequences,
                return_state = return_state,
                stateful=False,
                kernel_regularizer = tf.keras.regularizers.L1L2(l1=0.01, l2=0.01)
            )
          else:
            lstm = LSTM(
                self.hidden_layer_size,
                return_sequences=return_sequences,
                return_state = return_state,
                stateful=False,
                # Additional params to ensure LSTM matches CuDNN, See TF 2.0 :
                # (https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
                activation='tanh',
                recurrent_activation='sigmoid',
                recurrent_dropout=0,
                unroll=False,
                use_bias=True)
          return lstm
        historical_lstm, state_h, state_c = get_lstm(return_sequences = True, return_state = True)(historical_features)
        future_lstm = get_lstm(return_sequences = True, return_state = False)(future_features, initial_state=(state_h, state_c))
        encoder_features = tf.concat([historical_lstm, future_lstm], axis = 1)
        encoder_features = gating_layer(encoder_features, self.hidden_layer_size, self.dropout_rate)
        encoder_features = add_and_norm([skip, encoder_features])
        skip = encoder_features
        self_atten_layer = InterpretableMultiHeadAttention(self.num_heads,
                                                       self.hidden_layer_size,
                                                       dropout=self.dropout_rate)
        mask = get_decoder_mask(encoder_features)
        x, self_att = self_atten_layer(encoder_features, encoder_features, encoder_features, mask = mask)
        # x = x[Ellipsis, self.time_steps -  self.output_length: self.time_steps, :]
        x = gating_layer(x, self.hidden_layer_size, self.dropout_rate)
        x = add_and_norm([skip, x])

        x = TimeDistributed(Dense(1))(x[Ellipsis, self.time_steps -  self.output_length :, :])
        model = Model(inputs=[real_inputs, known_inputs], outputs=x)
        return model
        
    def build_model(self):
      model = self._build_base_model()
      
      adam = tf.keras.optimizers.Adam(learning_rate = self.learning_rate, clipnorm = self.max_gradient_norm)
      huber_loss = tf.keras.losses.Huber()
      model.compile(loss=huber_loss, optimizer=adam, sample_weight_mode='temporal')
      # tf.keras.utils.plot_model(model, './model.png', show_shapes=True)
      print(model.summary())
      return model
    
    def fit(self, train_dataset = None, valid_dataset = None):
      callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.early_stopping_patience,
            min_delta=1e-4),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=self.get_keras_saved_path(self._temp_folder, False),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True),
        tf.keras.callbacks.TerminateOnNaN()
      ]
      history = self.model.fit(
                train_dataset,
                epochs= self.num_epochs,
                callbacks = callbacks,
                batch_size=self.minibatch_size,
                validation_data = valid_dataset,
                use_multiprocessing = True
                )
      
      self.score_saving(history.history, self._score_folder)
      # Load best checkpoint again
      tmp_checkpont = self.get_keras_saved_path(self._temp_folder, True)
      if os.path.exists(tmp_checkpont):
        self.load(
            self._temp_folder,
            use_keras_loadings=True)

      else:
        print('Cannot load from {}, skipping ...'.format(self._temp_folder))
  # Serialisation.
    def reset_temp_folder(self):
      """Deletes and recreates folder with temporary Keras training outputs."""
      print('Resetting temp folder...')
      utils.create_folder_if_not_exist(self._temp_folder)
      shutil.rmtree(self._temp_folder)
      os.makedirs(self._temp_folder)

    def get_keras_saved_path(self, model_folder, to_check = False):
      """Returns path to keras checkpoint."""
      if to_check:
        return os.path.join(model_folder, '{}.check.index'.format(self.name))
      else:
        return os.path.join(model_folder, '{}.check'.format(self.name))

    def save(self, model_folder):
      """Saves optimal TFT weights.

      Args:
        model_folder: Location to serialze model.
      """
      # Allows for direct serialisation of tensorflow variables to avoid spurious
      # issue with Keras that leads to different performance evaluation results
      # when model is reloaded (https://github.com/keras-team/keras/issues/4875).

      self.model.save_weights(os.path.join(model_folder, '{}'.format(self.name)))
      print('Save model to {}'.format(os.path.join(model_folder, '{}'.format(self.name))))
      

    def load(self, model_folder, use_keras_loadings=False):
      """Loads TFT weights.

      Args:
        model_folder: Folder containing serialized models.
        use_keras_loadings: Whether to load from Keras checkpoint.

      Returns:

      """
      if use_keras_loadings:
        # Loads temporary Keras model saved during training.
        serialisation_path = self.get_keras_saved_path(model_folder, False)
        print('Loading model from {}'.format(serialisation_path))
        self.model.load_weights(serialisation_path)
      else:
        # Loads tensorflow graph for optimal models.
        print('Loading model from {}'.format(os.path.join(model_folder, '{}'.format(self.name))))
        self.model.load_weights(os.path.join(model_folder, '{}'.format(self.name)))

    def score_saving(self, history, model_folder):
      file_path = os.path.join(model_folder, "score.csv")
      if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        update_df = pd.DataFrame(history)
        df = pd.concat([df, update_df])
      else:
        df = pd.DataFrame(history)
      
      df.to_csv(file_path, index=False)


class ScaledDotProductAttention():
  """Defines scaled dot product attention layer.

  Attributes:
    dropout: Dropout rate to use
    activation: Normalisation function for scaled dot product attention (e.g.
      softmax by default)
  """

  def __init__(self, attn_dropout=0.0):
    self.dropout = Dropout(attn_dropout)
    self.activation = tf.keras.layers.Activation('softmax')

  def __call__(self, q, k, v, mask):
    """Applies scaled dot product attention.

    Args:
      q: Queries
      k: Keys
      v: Values
      mask: Masking if required -- sets softmax to very large value

    Returns:
      Tuple of (layer outputs, attention weights)
    """
    temper = tf.math.sqrt(tf.cast(tf.shape(k)[-1], dtype=tf.float32))
    attn = tf.linalg.matmul(q, k, transpose_b=True) / temper

    if mask is not None:
      mmask = tf.keras.layers.Lambda(lambda x: (-1e+9) * (1. - tf.keras.backend.cast(x, tf.float32)))(
          mask)  # setting to infinity
      attn = Add()([attn, mmask])
    attn = self.activation(attn)
    attn = self.dropout(attn)
    output = tf.keras.layers.Lambda(lambda x: tf.keras.backend.batch_dot(x[0], x[1]))([attn, v])
    return output, attn


class InterpretableMultiHeadAttention():
  """Defines interpretable multi-head attention layer.

  Attributes:
    n_head: Number of heads
    d_k: Key/query dimensionality per head
    d_v: Value dimensionality
    dropout: Dropout rate to apply
    qs_layers: List of queries across heads
    ks_layers: List of keys across heads
    vs_layers: List of values across heads
    attention: Scaled dot product attention layer
    w_o: Output weight matrix to project internal state to the original TFT
      state size
  """

  def __init__(self, n_head, d_model, dropout):
    """Initialises layer.

    Args:
      n_head: Number of heads
      d_model: TFT state dimensionality
      dropout: Dropout discard rate
    """

    self.n_head = n_head
    self.d_k = self.d_v = d_k = d_v = d_model // n_head
    self.dropout = dropout

    self.qs_layers = []
    self.ks_layers = []
    self.vs_layers = []

    # Use same value layer to facilitate interp
    vs_layer = Dense(d_v, use_bias=False)

    for _ in range(n_head):
      self.qs_layers.append(Dense(d_k, use_bias=False))
      self.ks_layers.append(Dense(d_k, use_bias=False))
      self.vs_layers.append(vs_layer)  # use same vs_layer

    self.attention = ScaledDotProductAttention()
    self.w_o = Dense(d_model, use_bias=False)

  def __call__(self, q, k, v, mask=None):
    """Applies interpretable multihead attention.

    Using T to denote the number of time steps fed into the transformer.

    Args:
      q: Query tensor of shape=(?, T, d_model)
      k: Key of shape=(?, T, d_model)
      v: Values of shape=(?, T, d_model)
      mask: Masking if required with shape=(?, T, T)

    Returns:
      Tuple of (layer outputs, attention weights)
    """
    n_head = self.n_head

    heads = []
    attns = []
    for i in range(n_head):
      qs = self.qs_layers[i](q)
      ks = self.ks_layers[i](k)
      vs = self.vs_layers[i](v)
      head, attn = self.attention(qs, ks, vs, mask)

      head_dropout = Dropout(self.dropout)(head)
      heads.append(head_dropout)
      attns.append(attn)
    head = tf.keras.backend.stack(heads) if n_head > 1 else heads[0]
    attn = tf.keras.backend.stack(attns)

    outputs = tf.keras.backend.mean(head, axis=0) if n_head > 1 else head
    outputs = self.w_o(outputs)
    outputs = Dropout(self.dropout)(outputs)  # output dropout

    return outputs, attn