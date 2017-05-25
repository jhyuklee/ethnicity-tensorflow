import tensorflow as tf
import os

from tensorflow.contrib.rnn.python.ops.rnn_cell import AttentionCellWrapper
from tensorflow.contrib.tensorboard.plugins import projector


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)


def lstm_cell(cell_dim, layer_num, keep_prob):
    with tf.variable_scope('LSTM_Cell') as scope:
        cell = tf.contrib.rnn.BasicLSTMCell(cell_dim, forget_bias=1.0, activation=tf.tanh, state_is_tuple=True)
        # cell = AttentionCellWrapper(cell, 10, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
        return tf.contrib.rnn.MultiRNNCell([cell] * layer_num, state_is_tuple=True)


def rnn_reshape(inputs, input_dim, max_time_step):
    with tf.variable_scope('Reshape') as scope:
        """
        reshape inputs from [batch_size, max_time_step, input_dim] to [max_time_step * (batch_size, input_dim)]

        :param inputs: inputs of shape [batch_size, max_time_step, input_dim]
        :param input_dim: dimension of input
        :param max_time_step: max of time step

        :return:
            outputs of shape [max_time_step * (batch_size, input_dim)]
        """
        inputs_tr = tf.transpose(inputs, [1, 0, 2])
        inputs_tr_reshape = tf.reshape(inputs_tr, [-1, input_dim])
        inputs_tr_reshape_split = tf.split(axis=0, num_or_size_splits=max_time_step,
                value=inputs_tr_reshape)
        return inputs_tr_reshape_split


def rnn_model(inputs, input_len, cell, params):
    max_time_step = params['max_time_step']
    dim_rnn_cell = params['dim_rnn_cell']
    with tf.variable_scope('RNN') as scope:
        outputs, state = tf.contrib.rnn.static_rnn(cell, inputs, sequence_length=input_len, dtype=tf.float32, scope=scope)
        outputs = tf.transpose(tf.stack(outputs), [1, 0, 2])
        spread_len = tf.range(0, tf.shape(input_len)[0]) * max_time_step + (input_len - 1)
        gathered_outputs = tf.gather(tf.reshape(outputs, [-1, dim_rnn_cell]), spread_len)
        return gathered_outputs


def bi_rnn_model(inputs, input_len, fw_cell, bw_cell):
    with tf.variable_scope('Bi-RNN') as scope:
        outputs, _, _ = tf.nn.bidirectional_rnn(fw_cell, bw_cell, inputs,
                sequence_length=input_len, dtype=tf.float32, scope=scope)
        outputs = tf.transpose(tf.pack(outputs), [1, 0, 2])
        return outputs


def embedding_lookup(inputs, voca_size, embedding_dim, visual_dir, config, draw=False,
        initializer=None, trainable=True, scope='Embedding'):
    with tf.variable_scope(scope) as scope:
        if initializer is not None:
            embedding_table = tf.get_variable("embed",
                    initializer=initializer, trainable=trainable, dtype=tf.float32)
        else:
            embedding_table = tf.get_variable("embed", [voca_size, embedding_dim],
                    dtype=tf.float32, trainable=trainable)
        inputs_embed = tf.nn.embedding_lookup(embedding_table, inputs)
        print(inputs_embed)

        if draw:
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_table.name
            embedding.metadata_path = os.path.join(visual_dir, '%s_metadata.tsv'%scope.name)
            return inputs_embed, projector
        else:
            return inputs_embed, None


def mask_by_index(batch_size, input_len, max_time_step):
    with tf.variable_scope('Masking') as scope:
        input_index = tf.range(0, batch_size) * max_time_step + (input_len - 1)
        lengths_transposed = tf.expand_dims(input_index, 1)
        lengths_tiled = tf.tile(lengths_transposed, [1, max_time_step])
        mask_range = tf.range(0, max_time_step)
        range_row = tf.expand_dims(mask_range, 0)
        range_tiled = tf.tile(range_row, [batch_size, 1])
        mask = tf.less_equal(range_tiled, lengths_tiled)
        weight = tf.select(mask, tf.ones([batch_size, max_time_step]),
                           tf.zeros([batch_size, max_time_step]))
        weight = tf.reshape(weight, [-1])
        return weight


def linear(inputs, output_dim, dropout_rate=1.0, regularize_rate=0, activation=None, scope='Linear'):
    with tf.variable_scope(scope) as scope:
        input_dim = inputs.get_shape().as_list()[-1]
        inputs = tf.reshape(inputs, [-1, input_dim])
        weights = tf.get_variable('Weights', [input_dim, output_dim],
                                  initializer=tf.random_normal_initializer())
        variable_summaries(weights, scope.name + '/Weights')
        biases = tf.get_variable('Biases', [output_dim],
                                 initializer=tf.constant_initializer(0.0))
        variable_summaries(biases, scope.name + '/Biases')
        if activation is None:
            return dropout((tf.matmul(inputs, weights) + biases), dropout_rate)
        else:
            return dropout(activation(tf.matmul(inputs, weights) + biases), dropout_rate)


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)
