import tensorflow as tf
from gensim.models.word2vec import Word2Vec
import numpy as np
import random

class TextCNN(object):
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      word_embedding_size, filter_sizes, num_filters, pretrained_embedding=True, 
      vocab=None, l2_reg_lambda=0.0, wordpos_embedding_dim=10,
      wordpos1_vocab_size=None, wordpos2_vocab_size=None):
       
 
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        
        
        # Placeholders for word pos vectors
        self.x_pos1 = tf.placeholder(tf.int32, [None, sequence_length], name="x_pos1")
        self.x_pos2 = tf.placeholder(tf.int32, [None, sequence_length], name="x_pos2")


        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.Variable(0.0, name="l2_loss")

        
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if not pretrained_embedding:
                W = tf.Variable(
                    tf.random_uniform([vocab_size, word_embedding_size], -1.0, 1.0),
                    name="W")
            else:
                model = Word2Vec.load_word2vec_format(
                        '/home/kungangli/Downloads/RE/PubMed-and-PMC-w2v.bin', binary=True)
                wordvecs = [[random.uniform(-1, 1) for _ in xrange(word_embedding_size)]] 
                for word in vocab:
                    try:
                        wordvec = model[word].tolist()
                    except KeyError: 
                        wordvec = [random.uniform(-1, 1) for _ in xrange(word_embedding_size)] 
                    wordvecs.append(wordvec)
                print wordvec
                print len(wordvec)
                W = tf.Variable(wordvecs, name="W") 
            W_pos1emb = tf.Variable(tf.random_uniform([wordpos1_vocab_size, wordpos_embedding_dim], -1.0, 1.0), 
                        name="W_pos1emb")
            W_pos2emb = tf.Variable(tf.random_uniform([wordpos2_vocab_size, wordpos_embedding_dim], -1.0, 1.0), 
                        name="W_pos2emb")
            self.word_embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.wordpos1_embedded_chars = tf.nn.embedding_lookup(W_pos1emb, self.x_pos1)
            self.wordpos2_embedded_chars = tf.nn.embedding_lookup(W_pos2emb, self.x_pos2)
            self.embedded_chars = tf.concat(2, [self.word_embedded_chars, 
                                  self.wordpos1_embedded_chars, self.wordpos2_embedded_chars])
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

            embedding_size = word_embedding_size + 2 * wordpos_embedding_dim

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)


        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])


        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)


        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")


        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss


        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

