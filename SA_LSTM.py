# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# set_session(tf.Session(config=config))

"""
Implementation of https://arxiv.org/pdf/1612.01887.pdf
"""
from keras import backend as K
from keras import activations
from keras import initializers
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops
from keras.engine.topology import Layer, InputSpec
from keras import regularizers
from keras import constraints
import tensorflow as tf

class SA_LSTM(Layer) :
    """
    input :
        feature matrix V_raw : (batch, location, filter)
    	max length scalar L : (batch, 1)
    output :
        probabilities throught time P : (batch, time, vocabsize)
    """
    def __init__(self,
                 vocabsize, units, middle_units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,

                 embeddings_output_size = 300,
                 embeddings_weights = None,
                 embeddings_trainable = None,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 embeddings_constraint=None,
                 implementation=2,
                 **kwargs):
        """
        vocabsize :
            vocabulary size.
        embeddings_weights :
            pretrained embedding matrix in numpy array.  
            could be Google word2vec or Stanford GloVe, or anything.
        embeddings_output_size :
            relevant only when embeddings_weights is not provided.
            new embeddings_weights will be created with size (vocabsize, embeddings_output_size).
        embeddings_trainable :
            relevant only when embeddings_weights is provided.
            whether embeddings_weights is trainable or not.
        """
        super(SA_LSTM, self).__init__(**kwargs)
        self.input_spec = [InputSpec(shape=(None, None, None)),
                           InputSpec(shape=(None, None))]
        self.vocabsize = vocabsize
        self.units = units
        self.middle_units = middle_units
        self.states = [None, None]
        self.implementation = implementation

        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        # embedding module
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        if (embeddings_weights is None) :
            if (embeddings_output_size is None) :
                raise ValueError("You shold provide embeddings_output_size "+
                                 "if embeddings_weights is not provied.")
            else :
                self.embeddings_output_size = embeddings_output_size
                embeddings_trainable = True
        else :
            if (len(embeddings_weights.shape) != 2) :
                raise ValueError("embeddings_weights must be a 2D numpy array, "+
                                 "found shape of "+str(embeddings_weights.shape)+".")
            if (embeddings_weights.shape[0] != vocabsize) :
                raise ValueError("embeddings_weights.shape[0] must equal to "+
                                 " vocab size = "+str(vocabsize)+
                                 ", found "+str(embeddings_weights.shape[0])+".")
            self.embeddings_output_size = embeddings_weights.shape[1]

        self.embeddings = self.add_weight(
            shape=(self.vocabsize, self.embeddings_output_size),
            initializer=self.embeddings_initializer,
            name='embeddings',
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            trainable=embeddings_trainable)
        if (embeddings_weights is not None) :
            K.set_value(self.embeddings, embeddings_weights)

    def build(self, input_shapes):
        """
        Variable names are based on the paper https://arxiv.org/pdf/1612.01887.pdf
        """
        self.input_spec[0] = InputSpec(shape=input_shapes[0])
        self.feature_shape = input_shapes[0][1:]
        self.locations = self.feature_shape[0]
        self.vector_size = self.feature_shape[1]

        # Attention kernels
        self.Wa = self.add_weight(
            shape=(self.vector_size, self.units),
            name='Wa',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.Wb = self.add_weight(
            shape=(self.vector_size, self.units),
            name='Wb',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.ba = self.add_weight(
            shape=(self.units,),
            name='ba',
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint)
        self.bb = self.add_weight(
            shape=(self.units,),
            name='bb',
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint)
        self.Wv = self.add_weight(
            shape=(self.units, self.middle_units),
            name='Wv',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.Wg = self.add_weight(
            shape=(self.units, self.middle_units),
            name='Wg',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.Ws = self.add_weight(
            shape=(self.units, self.middle_units),
            name='Ws',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.wh = self.add_weight(
            shape=(self.middle_units, 1),
            name='wh',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.Wp = self.add_weight(
            shape=(self.units, self.vocabsize),
            name='Wp',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)

        # LSTM kernels
        self.input_dim = self.embeddings_output_size + self.units
        self.kernel = self.add_weight(
            shape=(self.input_dim, self.units * 5),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 5),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(shape, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 3,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.units * 5,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.kernel_i = self.kernel[:, :self.units]
        self.kernel_f = self.kernel[:, self.units: self.units * 2]
        self.kernel_c = self.kernel[:, self.units * 2: self.units * 3]
        self.kernel_o = self.kernel[:, self.units * 3: self.units * 4]
        self.kernel_g = self.kernel[:, self.units * 4:]

        self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
        self.recurrent_kernel_f = self.recurrent_kernel[:, self.units: self.units * 2]
        self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2: self.units * 3]
        self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3: self.units * 4]
        self.recurrent_kernel_g = self.recurrent_kernel[:, self.units * 4:]

        if self.use_bias:
            self.bias_i = self.bias[:self.units]
            self.bias_f = self.bias[self.units: self.units * 2]
            self.bias_c = self.bias[self.units * 2: self.units * 3]
            self.bias_o = self.bias[self.units * 3: self.units * 4]
            self.bias_g = self.bias[self.units * 4:]
        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_c = None
            self.bias_o = None
            self.bias_g = None

        self.built = True

    def call(self, inputs):
        # in training mode, prev_words is the ground truth sentence.
        # in testing mode, prev_words is just a dummy tensor.
        feature_vectors, prev_words = inputs

        self.ag = K.mean(feature_vectors, axis=1)

        self.V = K.relu(K.bias_add(K.dot(feature_vectors, self.Wa), self.ba), alpha=0.2)
        self.vg = K.relu(K.bias_add(K.dot(self.ag, self.Wb), self.bb), alpha=0.2)

        # self.V = K.relu(K.dot(feature_vectors, self.Wa), alpha=0.2)
        # self.vg = K.relu(K.dot(self.ag, self.Wb), alpha=0.2)

        self.WvV = K.dot(self.V, self.Wv)

        prev_words = K.cast(tf.transpose(prev_words, [1,0]), "int64")
        time_steps = tf.shape(prev_words)[0]

        input_ta = tensor_array_ops.TensorArray(
            dtype=prev_words.dtype,
            size=time_steps).unstack(prev_words)

        output_ta = tensor_array_ops.TensorArray(
            dtype = feature_vectors.dtype,
            clear_after_read=False,
            size=time_steps)

        padding = [[0,0], [self.embeddings_output_size, 0]]
        padding = tf.convert_to_tensor(padding)
        first_inputs = tf.pad(self.vg, paddings=padding)
        states = self.get_initial_state(feature_vectors)
        step_function = self.step
        first_prob, states = step_function(first_inputs, states)
        output_ta = output_ta.write(0, first_prob)

        # start at t=1
        prev_time = tf.constant(0, dtype='int32')
        cur_time = tf.constant(1, dtype='int32')
        learning_phase = K.learning_phase()

        def _step(prev_time, cur_time, output_ta_t, *states) :
            train_phase_indices = input_ta.read(prev_time)

            prob = output_ta_t.read(prev_time)
            test_phase_indices = K.argmax(K.stop_gradient(prob), axis=-1)
            indices = K.in_train_phase(train_phase_indices, test_phase_indices, learning_phase)

            embeddings_vector = K.gather(self.embeddings, indices)
            inputs = K.concatenate([embeddings_vector, self.vg], axis=-1)
            output, new_states = step_function(inputs, tuple(states))
            for state, new_state in zip(states, new_states):
                new_state.set_shape(state.get_shape())
            output_ta_t = output_ta_t.write(cur_time, output)
            return (cur_time, cur_time+1, output_ta_t) + tuple(new_states)

        final_outputs = control_flow_ops.while_loop(
            cond=lambda prev_time, cur_time, *_: cur_time < time_steps,
            body=_step,
            loop_vars=(prev_time, cur_time, output_ta) + states,
            parallel_iterations=32,
            swap_memory=True)

        # last_time = final_outputs[0]
        output_ta = final_outputs[2]
        outputs = output_ta.stack()
        axes = [1, 0] + list(range(2, len(outputs.get_shape())))
        outputs = tf.transpose(outputs, axes)
        outputs._uses_learning_phase = True

        return outputs

    def step(self, inputs, states):
        """
        LSTM step + attention
        inputs : concatenate of embedding vector and mean of feature vectors
        """

        #LSTM phase

        h_tm1 = states[0]
        c_tm1 = states[1]

        if self.implementation == 2:
            z = K.dot(inputs, self.kernel)
            z += K.dot(h_tm1, self.recurrent_kernel)
            if self.use_bias:
                z = K.bias_add(z, self.bias)

            zi = z[:, :self.units]
            zf = z[:, self.units: 2 * self.units]
            zc = z[:, 2 * self.units: 3 * self.units]
            zo = z[:, 3 * self.units: 4 * self.units]
            zg = z[:, 4 * self.units:]

            i = self.recurrent_activation(zi)
            f = self.recurrent_activation(zf)
            c = f * c_tm1 + i * self.activation(zc)
            o = self.recurrent_activation(zo)
            g = self.recurrent_activation(zg)
        else:
            if self.implementation == 0:
                raise ValueError('Keras RNN implementation 0 is not allowed for this layer.')
            elif self.implementation == 1:
                x_i = K.dot(inputs, self.kernel_i) + self.bias_i
                x_f = K.dot(inputs, self.kernel_f) + self.bias_f
                x_c = K.dot(inputs, self.kernel_c) + self.bias_c
                x_o = K.dot(inputs, self.kernel_o) + self.bias_o
                x_g = K.dot(inputs, self.kernel_g) + self.bias_g
            else:
                raise ValueError('Unknown `implementation` mode.')

            i = self.recurrent_activation(x_i + K.dot(h_tm1, self.recurrent_kernel_i))
            f = self.recurrent_activation(x_f + K.dot(h_tm1, self.recurrent_kernel_f))
            c = f*c_tm1 + i*self.activation(x_c + K.dot(h_tm1, self.recurrent_kernel_c))
            o = self.recurrent_activation(x_o + K.dot(h_tm1, self.recurrent_kernel_o))
            g = self.recurrent_activation(x_g + K.dot(h_tm1, self.recurrent_kernel_g))

        h = o * self.activation(c)
        s = g * self.activation(c)

        # attention phase
        h1 = K.expand_dims(h, axis=1)
        s1 = K.expand_dims(s, axis=1)
        h2 = K.dot(h1, self.Wg)
        s2 = K.dot(s1, self.Ws)

        V_old = K.concatenate([self.V, s1], axis=1)
        V_new = K.concatenate([self.WvV, s2], axis=1)

        combine = tf.tanh(tf.add(V_new, h2))
        z = K.dot(combine, self.wh)
        print "Z :", z.shape
        alpha = tf.nn.softmax(z, dim=1)
        print "alpha :", alpha.shape

        c_hat = K.sum(tf.multiply(alpha, V_old), axis=1)

        result = K.softmax(K.dot(c_hat + h, self.Wp))

        return result, (h, c)

    def get_initial_state(self, inputs):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(inputs)  # (samples, location, filter)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        initial_state = K.tile(initial_state, [1, self.units])  # (samples, output_dim)
        initial_state = [initial_state for _ in range(len(self.states))]
        return initial_state

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], None) + (self.vocabsize,)

# Test
if __name__ == "__main__" :
    vocab = 5000
    loc = 5*5
    fea = 20
    units = 120
    middle_units = 100
    emb = 300
    max_len = 18
    batchsize = 100

    from keras.layers import Input
    from keras.models import Model
    import numpy as np

    inp = Input((loc, fea))
    P = Input((None,))

    emb_mat = np.load("embedding/matrix.npy")
    lstm = SA_LSTM(
        vocabsize=vocab,
        units=units, middle_units=middle_units,
        embeddings_weights=emb_mat,
        embeddings_trainable=True)
    x = lstm([inp, P])

    e = lstm.embeddings

    model = Model([inp, P], x)
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["acc"])
    model.summary()
    print model.uses_learning_phase
    
    X = np.random.uniform(-1, 1, (batchsize, loc, fea))
    YP = np.ones((batchsize, max_len))
    Y = np.random.randint(vocab, size=(batchsize, max_len, 1))

    print X.shape, Y.shape

    model.fit([X, YP], Y, epochs=5)
    model.predict([X, YP])

    new_mat = K.eval(e)