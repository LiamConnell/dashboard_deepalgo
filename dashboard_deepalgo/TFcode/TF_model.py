import tensorflow as tf
import numpy as np
import numpy.random as rng
import pandas.io.data as web
import numpy as np
import pandas as pd

class Model():
    def __init__(self, config, training=True):
        #CONFIG
        symbol_list = self.symbol_list = config.symbol_list
        num_samples = self.num_samples =  config.num_samples
        input_len = self.input_len =  config.input_len
        n_hidden_1 = self.n_hidden_1 =  config.n_hidden_1
        n_hidden_2 = self.n_hidden_2 =  config.n_hidden_2 
        learning_rate = self.learning_rate = config.learning_rate
        
        #bucket info
        positions = self.positions = tf.constant([-1,0,1])
        num_positions = self.num_positions =  3
        
        #more vars
        num_symbols = self.num_symbols =  len(symbol_list)
        n_input = self.n_input = num_symbols * input_len
        n_classes = self.n_classes = num_positions * num_symbols 

        
        
        x =self.x = tf.placeholder(tf.float32, [None, n_input])
        y_ =self.y_= tf.placeholder(tf.float32, [None,  num_symbols])

        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        
        def multilayer_perceptron(x, weights, biases, keep_prob):
            # Hidden layer with RELU activation
            layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
            layer_1 = tf.nn.relu(layer_1)
            # Hidden layer with RELU activation
            layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
            layer_2 = tf.nn.relu(layer_2)
            # Output layer with linear activation
            out_layer_f = tf.matmul(layer_2, weights['out']) + biases['out']
            out_layer = tf.nn.dropout(out_layer_f, keep_prob)                 # DROPOUT LAYER
            return out_layer
        
        if training == True: keep_prob = 0.5  # DROPOUT
        else: keep_prob = 1.0                 # NO DROPOUT
            
        # Construct model
        y = multilayer_perceptron(x, weights, biases, keep_prob)



        # loop through symbol, taking the columns for each symbol's bucket together
        pos = {}
        sample_n = {}
        sample_mask = {}
        symbol_returns = {}
        relevant_target_column = {}
        for i in range(num_symbols):
            # isolate the buckets relevant to the symbol and get a softmax as well
            symbol_probs = y[:,i*num_positions:(i+1)*num_positions]
            symbol_probs_softmax = tf.nn.softmax(symbol_probs) # softmax[i, j] = exp(logits[i, j]) / sum(exp(logits[i]))
            # sample probability to chose our policy's action
            sample = tf.multinomial(tf.log(symbol_probs_softmax), num_samples)
            for sample_iter in range(num_samples):
                sample_n[i*num_samples + sample_iter] = sample[:,sample_iter]
                pos[i*num_samples + sample_iter] = tf.reshape(sample_n[i*num_samples + sample_iter], [-1]) - 1
                symbol_returns[i*num_samples + sample_iter] = tf.mul(
                                                                    tf.cast(pos[i*num_samples + sample_iter], tf.float32), 
                                                                     y_[:,i])

                sample_mask[i*num_samples + sample_iter] = tf.cast(tf.reshape(tf.one_hot(sample_n[i*num_samples + sample_iter], 3), [-1,3]), tf.float32)
                relevant_target_column[i*num_samples + sample_iter] = tf.reduce_sum(
                                                            symbol_probs_softmax * sample_mask[i*num_samples + sample_iter],1)
        self.pos = pos

        # PERFORMANCE METRICS
        daily_returns_by_symbol_ = tf.concat(1, [tf.reshape(t, [-1,1]) for t in symbol_returns.values()])
        daily_returns_by_symbol = tf.transpose(tf.reshape(daily_returns_by_symbol_, [-1,2,num_samples]), [0,2,1]) #[?,5,2]
        self.daily_returns = daily_returns = tf.reduce_mean(daily_returns_by_symbol, 2) # [?,5]
        
        total_return = tf.reduce_prod(daily_returns+1, 0)
        z = tf.ones_like(total_return) * -1
        self.total_return =total_return= tf.add(total_return, z)
        
        self.ann_vol = ann_vol = tf.mul(
            tf.sqrt(tf.reduce_mean(tf.pow((daily_returns - tf.reduce_mean(daily_returns, 0)),2),0)) ,
            np.sqrt(252)
            )
        self.sharpe = sharpe = tf.div(total_return, ann_vol)
        #Maybe metric slicing later
        #segment_ids = tf.ones_like(daily_returns[:,0])
        #partial_prod = tf.segment_prod(daily_returns+1, segment_ids)


        training_target_cols = tf.concat(1, [tf.reshape(t, [-1,1]) for t in relevant_target_column.values()])
        ones = tf.ones_like(training_target_cols)
        gradient_ = tf.nn.sigmoid_cross_entropy_with_logits(training_target_cols, ones)
        gradient = tf.transpose(tf.reshape(gradient_, [-1,2,num_samples]), [0,2,1]) #[?,5,2]

        #L2  = tf.contrib.layers.l2_regularizer(0.1)
        #t_vars = tf.trainable_variables()
        #reg = tf.contrib.layers.apply_regularization(L2, tf.GraphKeys.WEIGHTS)

        #cost = tf.mul(gradient , daily_returns_by_symbol_reshaped)
        #cost = tf.mul(gradient , tf.expand_dims(daily_returns, -1)) #+ reg
        #cost = tf.mul(gradient , tf.expand_dims(total_return, -1))
        cost = tf.mul(gradient , tf.expand_dims(sharpe, -1))

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        self.costfn = tf.reduce_mean(cost)