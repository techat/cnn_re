import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import dataloader
from text_cnn import TextCNN
import datetime
import time
import os
from sklearn.model_selection import KFold

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("data_file", "/home/kungangli/Downloads/RE/GAD_Corpus_IBIgroup/train.csv", "Data source")
tf.flags.DEFINE_string("delimiter", "\t", "Delimiter in the data file")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_bool("pretrained_word2vec", False, "Use pre-trained word2vec vectors to initialize word embeddings (default: False)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================

# Load data
x_text, entity1, entity2, y = dataloader.load_data_and_labels(FLAGS.data_file, FLAGS.delimiter)
print x_text[0:5], entity1[0:5], entity2[0:5], y[0:5]

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Extract word:id mapping from the object.
vocab_dict = vocab_processor.vocabulary_._mapping

# Sort the vocabulary dictionary on the basis of values(id).
# Both statements perform same task.
#sorted_vocab = sorted(vocab_dict.items(), key=operator.itemgetter(1))
sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1])

# Treat the id's as index into list and create a list of words in the ascending order of id's
# word with id i goes at index i of the list.
vocabulary = list(list(zip(*sorted_vocab))[0])


# kfold CV
trains, devs = [], []
kf = KFold(n_splits=4, shuffle=True, random_state=10)
for train_index, test_index in kf.split(x):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_dev = x[train_index], x[test_index]
    y_train, y_dev = y[train_index], y[test_index]
    trains.append((x_train, y_train))
    devs.append((x_dev, y_dev))


# Training
# ==================================================
test_accuracies = []

for ind in xrange(len(trains)):
    print("======================================================")
    print("The first run in KFold CV: ", ind)
    x_train, y_train = trains[ind]
    x_dev, y_dev = devs[ind]
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
              allow_soft_placement=FLAGS.allow_soft_placement,
              log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
	    cnn = TextCNN(
		sequence_length=x_train.shape[1],
		num_classes=y_train.shape[1],
		vocab_size=len(vocab_processor.vocabulary_),
		embedding_size=FLAGS.embedding_dim,
		filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
		num_filters=FLAGS.num_filters,
		pretrained_embedding=FLAGS.pretrained_word2vec,
		vocab=vocabulary,
		l2_reg_lambda=FLAGS.l2_reg_lambda)

	    # Define Training procedure
	    global_step = tf.Variable(0, name="global_step", trainable=False)
	    optimizer = tf.train.AdamOptimizer(1e-3)
	    grads_and_vars = optimizer.compute_gradients(cnn.loss)
	    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

	    # Keep track of gradient values and sparsity (optional)
	    grad_summaries = []
	    for g, v in grads_and_vars:
		if g is not None:
		    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
		    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
		    grad_summaries.append(grad_hist_summary)
		    grad_summaries.append(sparsity_summary)
	    grad_summaries_merged = tf.summary.merge(grad_summaries)

	    # Output directory for models and summaries
	    timestamp = str(int(time.time()))
	    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
	    print("Writing to {}\n".format(out_dir))

	    # Summaries for loss and accuracy
	    loss_summary = tf.summary.scalar("loss", cnn.loss)
	    acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

	    # Train Summaries
	    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
	    train_summary_dir = os.path.join(out_dir, "summaries", "train")
	    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

	    # Dev summaries
	    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
	    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
	    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

	    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
	    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
	    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
	    if not os.path.exists(checkpoint_dir):
		os.makedirs(checkpoint_dir)
	    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

	    # Write vocabulary
	    vocab_processor.save(os.path.join(out_dir, "vocab"))

	    # Initialize all variables
	    sess.run(tf.global_variables_initializer())

	    def train_step(x_batch, y_batch):
		"""
		A single training step
		"""
		feed_dict = {
		  cnn.input_x: x_batch,
		  cnn.input_y: y_batch,
		  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
		}
		_, step, summaries, loss, accuracy = sess.run(
		    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
		    feed_dict)
		time_str = datetime.datetime.now().isoformat()
		print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
		train_summary_writer.add_summary(summaries, step)

	    def dev_step(x_batch, y_batch, writer=None):
		"""
		Evaluates model on a dev set
		"""
		feed_dict = {
		  cnn.input_x: x_batch,
		  cnn.input_y: y_batch,
		  cnn.dropout_keep_prob: 1.0
		}
		step, summaries, loss, accuracy = sess.run(
		    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
		    feed_dict)
		time_str = datetime.datetime.now().isoformat()
		print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                test_accuracies.append((ind, step, accuracy)) 
		if writer:
		    writer.add_summary(summaries, step)

	    # Generate batches
	    batches = dataloader.batch_iter(
		list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
	    # Training loop. For each batch...
	    for batch in batches:
		x_batch, y_batch = zip(*batch)
		train_step(x_batch, y_batch)
		current_step = tf.train.global_step(sess, global_step)
		if current_step % FLAGS.evaluate_every == 0:
		    print("\nEvaluation:")
		    dev_step(x_dev, y_dev, writer=dev_summary_writer)
		    print("")
		if current_step % FLAGS.checkpoint_every == 0:
		    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
		    print("Saved model checkpoint to {}\n".format(path))
    tf.reset_default_graph()

print test_accuracies
