import tensorflow as tf
import os

from ops import *


class RNN(object):
    def __init__(self, params, initializer):

        # session settings
        config = tf.ConfigProto(device_count={'GPU':1})
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.session = tf.Session(config=config)
        self.params = params
        self.model_name = params['model_name']

        # hyper parameters
        self.learning_rate = params['learning_rate']
        self.decay_rate = params['decay_rate']
        self.decay_step = params['decay_step']
        self.min_grad = params['min_grad']
        self.max_grad = params['max_grad']

        # rnn parameters
        self.max_time_step = params['max_time_step']
        self.cell_layer_num = params['lstm_layer']
        self.dim_embed_unigram = params['dim_embed_unigram']
        self.dim_embed_bigram = params['dim_embed_bigram']
        self.dim_embed_trigram = params['dim_embed_trigram']
        self.dim_hidden = params['dim_hidden']
        self.dim_rnn_cell = params['dim_rnn_cell']
        self.dim_unigram = params['dim_unigram'] 
        self.dim_bigram = params['dim_bigram'] 
        self.dim_trigram = params['dim_trigram'] 
        self.dim_output = params['dim_output']
        self.ngram = params['ngram']
        self.ensemble = params['ensemble']
        self.embed = params['embed']
        self.embed_trainable = params['embed_trainable']
        self.checkpoint_dir = params['checkpoint_dir']
        self.initializer = initializer

        # input data placeholders
        self.unigram = tf.placeholder(tf.int32, [None, self.max_time_step])
        self.bigram = tf.placeholder(tf.int32, [None, self.max_time_step])
        self.trigram = tf.placeholder(tf.int32, [None, self.max_time_step])
        self.lengths = tf.placeholder(tf.int32, [None])
        self.labels = tf.placeholder(tf.int32, [None])
        self.lstm_dropout = tf.placeholder(tf.float32)
        self.hidden_dropout = tf.placeholder(tf.float32)

        # model settings
        self.global_step = tf.Variable(0, name="step", trainable=False)
        self.learning_rate = tf.train.exponential_decay(
                self.learning_rate, self.global_step,
                self.decay_step, self.decay_rate, staircase=True)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = None
        self.saver = None
        self.losses = None
        self.logits = None

        # model build
        self.merged_summary = None
        self.embed_writer = tf.summary.FileWriter(self.checkpoint_dir)
        self.embed_config = projector.ProjectorConfig()
        self.projector = None
        self.build_model()
        self.session.run(tf.global_variables_initializer())
       
        # debug initializer
        '''
        with tf.variable_scope('Unigram', reuse=True):
            unigram_embed = tf.get_variable("embed", [self.dim_unigram, self.dim_embed_unigram], dtype=tf.float32)
            print(unigram_embed.eval(session=self.session))
        '''

    def ngram_logits(self, inputs, length, dim_input, dim_embed=None, 
            initializer=None, trainable=True, scope='ngram'):
        with tf.variable_scope(scope) as scope: 
            fw_cell = lstm_cell(self.dim_rnn_cell, self.cell_layer_num, self.lstm_dropout)
            bw_cell = lstm_cell(self.dim_rnn_cell, self.cell_layer_num, self.lstm_dropout)
            
            if dim_embed is not None:
                inputs_embed, self.projector = embedding_lookup(inputs, 
                        dim_input, dim_embed, self.checkpoint_dir, self.embed_config, 
                        draw=True, initializer=initializer, trainable=trainable, scope=scope)
                inputs_reshape = rnn_reshape(inputs_embed, dim_embed, self.max_time_step)
                self.projector.visualize_embeddings(self.embed_writer, self.embed_config)
            else:
                inputs_reshape = rnn_reshape(tf.one_hot(inputs, dim_input), dim_input, self.max_time_step)
            
            outputs = rnn_model(inputs_reshape, length, fw_cell, self.params)
            return outputs

    def build_model(self):
        print("## Building an RNN model")

        unigram_logits = self.ngram_logits(inputs=self.unigram, 
                length=self.lengths, 
                dim_input=self.dim_unigram,
                dim_embed=self.dim_embed_unigram if self.embed else None,
                initializer=self.initializer[0],
                trainable=self.embed_trainable,
                scope='Unigram')

        bigram_logits = self.ngram_logits(inputs=self.bigram, 
                length=self.lengths-1, 
                dim_input=self.dim_bigram,
                dim_embed=self.dim_embed_bigram if self.embed else None,
                initializer=self.initializer[1],
                trainable=self.embed_trainable,
                scope='Bigram')
        
        trigram_logits = self.ngram_logits(inputs=self.trigram, 
                length=self.lengths-2, 
                dim_input=self.dim_trigram,
                dim_embed=self.dim_embed_trigram if self.embed else None,
                initializer=self.initializer[2],
                trainable=self.embed_trainable,
                scope='Trigram')

        if self.ensemble:
            total_logits = tf.concat([unigram_logits, bigram_logits, trigram_logits], axis=1)
        elif self.ngram == 1:
            total_logits = unigram_logits
        elif self.ngram == 2:
            total_logits = bigram_logits
        elif self.ngram == 3:
            total_logits = trigram_logits
        else:
            assert True, 'No specific ngram %d'% ngram

        hidden1 = linear(inputs=total_logits, 
                output_dim=self.dim_hidden,
                dropout_rate=self.hidden_dropout,
                activation=tf.nn.relu,
                scope='Hidden1')
        
        logits = linear(inputs=total_logits,
            output_dim=self.dim_output, 
            scope='Output')

        self.logits = logits 
        self.losses = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
            labels=self.labels))

        tf.summary.scalar('Loss', self.losses)
        self.variables = tf.trainable_variables()

        grads = []
        for grad in tf.gradients(self.losses, self.variables):
            if grad is not None:
                grads.append(tf.clip_by_value(grad, self.min_grad, self.max_grad))
            else:
                grads.append(grad)
        self.optimize = self.optimizer.apply_gradients(zip(grads, self.variables), global_step=self.global_step)

        model_vars = [v for v in tf.global_variables()]
        print('model variables', [model_var.name for model_var in tf.trainable_variables()])
        self.saver = tf.train.Saver(model_vars)
        self.merged_summary = tf.summary.merge_all()

    @staticmethod
    def reset_graph():
        tf.reset_default_graph()

    def save(self, checkpoint_dir, step):
        file_name = "%s.model" % self.model_name
        self.saver.save(self.session, os.path.join(checkpoint_dir, file_name))
        print("Model saved", file_name)

    def load(self, checkpoint_dir):
        file_name = "%s.model" % self.model_name
        file_name += "-10800"
        self.saver.restore(self.session, os.path.join(checkpoint_dir, file_name))
        print("Model loaded", file_name)

