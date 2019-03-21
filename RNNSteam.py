import numpy as np
import tensorflow as tf
from collections import Counter
from string import punctuation

def make_rnn(reviews_txt_file, scores_txt_file, training_percent, saved_name):
    with open(reviews_txt_file,'r') as f:
        reviews = f.read()
    with open(scores_txt_file,'r') as f:
        scores = f.read()

    reviews_text = ''.join([c for c in reviews if c not  in punctuation])
    reviews = reviews_text.split('\n')

    words_text = ' '.join(reviews)
    words = words_text.split()

    word_counts = Counter(words)
    vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

    reviews_ints = []
    for review in reviews:
        reviews_ints.append([vocab_to_int[word] for word in review.split()])

    scores = scores.split('\n')
    scores = np.array([1 if score == '1' else 0 for score in scores])

    review_lens = Counter([len(x) for x in reviews_ints])

    non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]

    reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]
    scores = np.array([scores[ii] for ii in non_zero_idx])

    sequence_length = 200
    layer_0 = np.zeros((len(reviews_ints), sequence_length), dtype=int)
    for i, row in enumerate(reviews_ints):
        layer_0[i, -len(row):] = np.array(row)[:sequence_length]

    review_training_split_index = int(len(layer_0)*training_percent)
    review_training_set, review_validation_set = layer_0[:review_training_split_index], layer_0[review_training_split_index:]
    score_training_set, score_validation_set = scores[:review_training_split_index], scores[review_training_split_index:]

    review_testing_split_index = int(len(review_validation_set)*0.5)
    review_validation_set, review_test_set = review_validation_set[:review_testing_split_index], review_validation_set[review_testing_split_index:]
    score_validation_set, score_test_set = score_validation_set[:review_testing_split_index], score_validation_set[review_testing_split_index:]

    lstm_size = 256
    lstm_layers = 1
    batch_size = 512
    learning_rate = 0.001

    number_of_words = len(vocab_to_int) + 1

    # Create the graph object
    graph = tf.Graph()
    # Add nodes to the graph
    with graph.as_default():
        inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
        scores_ = tf.placeholder(tf.int32, [None, None], name='scores')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    embedded_layer_size = 300 

    with graph.as_default():
        embedding = tf.Variable(tf.random_uniform((number_of_words, embedded_layer_size), -1, 1))
        embed = tf.nn.embedding_lookup(embedding, inputs_)

    with graph.as_default():
        # Your basic LSTM cell
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        
        # Add dropout to the cell
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        
        # Stack up multiple LSTM layers, for deep learning
        cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
        
        # Getting an initial state of all zeros
        initial_state = cell.zero_state(batch_size, tf.float32)

    with graph.as_default():
        outputs, final_state = tf.nn.dynamic_rnn(cell, embed,
                                                initial_state=initial_state)

    with graph.as_default():
        predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
        cost = tf.losses.mean_squared_error(scores_, predictions)
        
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    with graph.as_default():
        correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), scores_)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def get_batches(x, y, batch_size=100):
        
        n_batches = len(x)//batch_size
        x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
        for ii in range(0, len(x), batch_size):
            yield x[ii:ii+batch_size], y[ii:ii+batch_size]

    epochs = 1

    with graph.as_default():
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        iteration = 1
        for e in range(epochs):
            state = sess.run(initial_state)
            
            for ii, (x, y) in enumerate(get_batches(review_training_set, score_training_set, batch_size), 1):
                feed = {inputs_: x,
                        scores_: y[:, None],
                        keep_prob: 0.5,
                        initial_state: state}
                loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)
                
                if iteration%5==0:
                    print("Epoch: {}/{}".format(e, epochs),
                        "Iteration: {}".format(iteration),
                        "Train loss: {:.3f}".format(loss))

                if iteration%25==0:
                    val_acc = []
                    val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                    for x, y in get_batches(review_validation_set, score_validation_set, batch_size):
                        feed = {inputs_: x,
                                scores_: y[:, None],
                                keep_prob: 1,
                                initial_state: val_state}
                        batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                        val_acc.append(batch_acc)
                    print("Val acc: {:.3f}".format(np.mean(val_acc)))
                iteration +=1
        saver.save(sess, "checkpoints/" + saved_name + ".ckpt")


    test_acc = []
    with tf.Session(graph=graph) as sess:
        saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
        test_state = sess.run(cell.zero_state(batch_size, tf.float32))
        for ii, (x, y) in enumerate(get_batches(review_test_set, score_test_set, batch_size), 1):
            feed = {inputs_: x,
                    scores_: y[:, None],
                    keep_prob: 1,
                    initial_state: test_state}
            batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
            test_acc.append(batch_acc)
        print("Test accuracy: {:.3f}".format(np.mean(test_acc)))

    return np.mean(test_acc)

