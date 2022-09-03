from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Dense, Bidirectional, GRU, LSTM, Embedding

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as k
from tensorflow.keras import initializers, regularizers, constraints


class Attention(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias

        self.W = None
        self.b = None
        self.u = None
        self.a = None

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(Attention, self).build(input_shape)

    def compute_mask(self, x, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        a = self.get_attention_weights(x, mask)
        a = k.expand_dims(a)
        weighted_input = x * a
        self.a = a
        return k.sum(weighted_input, axis=1)

    def get_attention_weights(self, x, mask=None):
        # ut = tanh(Ww ht + bw) where ut is the hidden representation of the hidden state ht
        ut = k.squeeze(k.dot(x, k.expand_dims(self.W)), axis=-1)
        if self.bias:
            ut += self.b
        ut = k.tanh(ut)

        # Multiply hidden representation by word-level context vector (uw)
        at = k.squeeze(k.dot(ut, k.expand_dims(self.u)), axis=-1)

        # Apply softmax to normalise
        a = k.exp(at)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= k.cast(mask, k.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= k.cast(k.sum(a, axis=1, keepdims=True) + k.epsilon(), k.floatx())

        return a

    @staticmethod
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_config(self):
        return super(Attention, self).get_config()


class Sum(Layer):
    """Custom layer that just does a simple sum of the vector input, not sure why keras doesnt have it"""
    def __init__(self, **kwargs):
        super(Sum, self).__init__(**kwargs)

    @staticmethod
    def call(inputs):
        return k.sum(inputs, axis=1)

    @staticmethod
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_config(self):
        return super(Sum, self).get_config()


def get_layer(layer, rnn, bi, att):
    """Returns the main layer in the saved_models. The following layers are trained in our paper:
        1. Dense (baseline)
        2. LSTM
        3. GRU
        4. BiLSTM
        5. BiGRU
        6. BiLSTM with attention
        7. BiGRU with attention
    """

    if rnn == "baseline":  # Dense
        add_layer = Sum()(layer)
        return Dense(64, activation='relu')(add_layer)
    elif rnn == "lstm":
        rnn_layer = LSTM(64, dropout=0.1)
    else:
        rnn_layer = GRU(64, dropout=0.1)

    if bi == "":  # LSTM/GRU
        return rnn_layer(layer)

    if att == "":  # BiLSTM/BiGRU
        rnn_layer = Bidirectional(rnn_layer)
        return rnn_layer(layer)

    # BiLSTM/BiGRU with attention
    rnn_layer.return_sequences = True
    rnn_layer = Bidirectional(rnn_layer)(layer)
    attention_layer = Attention()(rnn_layer)

    return attention_layer


def get_model(train, rnn, bi, emb, att, q):
    """Returns a neural network model based on hyperparameters
        train: whether to train or freeze the embeddings
        rnn: whether to use a dense, lstm or gru layer
        bi: whether the lstm/gru layer is bidirectional
        emb: type of embedding (GloVe, fastText or lda2vec)
        att: whether the lstm/gru has attention mechanism
        q: question number
    """

    input_layer = Input(shape=(None,))

    embedding_layer = Embedding(
        input_dim=len(emb),
        output_dim=300,
        weights=[emb],
        trainable=train,
        mask_zero=True)(input_layer)

    rnn_layer = get_layer(embedding_layer, rnn, bi, att)

    dense_layer1 = Dense(64, activation='relu')(rnn_layer)

    # Since q1 has 3 possible values for output and q2 has 4 possible values for output, we covneniently use
    # the question number to give the shape for the output layer
    dense_layer2 = Dense(2 + int(q), activation='softmax')(dense_layer1)

    model = Model(inputs=input_layer, outputs=dense_layer2)

    # We use Adam optimizer because it's one of the better ones
    opt = Adam(learning_rate=0.001)

    # Compile the model
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
