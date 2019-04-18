import numpy as np
import tensorflow as tf
from collections import Counter
from string import punctuation

def make_rnn(reviews_txt_file, scores_txt_file, input_review, input_score, game):
    
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

    lstm_size = 256
    lstm_layers = 1
    batch_size = 1
    learning_rate = 0.001

    number_of_words = len(vocab_to_int) + 1

    graph = tf.Graph()

    with graph.as_default():
        inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
        print(inputs_)
        scores_ = tf.placeholder(tf.int32, [None, None], name='scores')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    embedded_layer_size = 300 

    with graph.as_default():
        embedding = tf.Variable(tf.random_uniform((number_of_words, embedded_layer_size), -1, 1))
        embed = tf.nn.embedding_lookup(embedding, inputs_)

    with graph.as_default():

        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
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

    test_acc = []
    with tf.Session(graph=graph) as sess:

        input_review_int_array = np.zeros((1, 200), dtype=int)
        # int_array = [vocab_to_int[word] for word in input_review.split()]
        int_array = []
        for word in input_review.split():
            if word in vocab_to_int:
                int_array.append(vocab_to_int[word])
        print(int_array)
        input_review_int_array[0, -len(int_array):] = np.array(int_array)[:200]
        input_score_int_array =  np.array([input_score])
    
        saver = tf.train.import_meta_graph('checkpoints/' + game + '.ckpt.meta')
        saver.restore(sess, 'checkpoints/' + game + '.ckpt')
        print('1')
        test_state = sess.run(cell.zero_state(batch_size, tf.float32))
        for ii, (x, y) in enumerate(get_batches(input_review_int_array, input_score_int_array, batch_size), 1):
            feed = {inputs_: x,
                    scores_: y[:, None],
                    keep_prob: 1,
                    initial_state: test_state}
            batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
            test_acc.append(batch_acc)

        print("Test accuracy: {:.3f}".format(np.mean(test_acc)))
        if np.mean(test_acc) == 1.0:
            return True
        else:
            return False

# batmanarkhamnight_accuracy = make_rnn('dota2splitreviews.txt', 'dota2splitscores.txt', .8, 'dota2split')

print(make_rnn('dota2splitreviews.txt', 'dota2splitscores.txt', "i love this dfsgisdmgf year game", .8, 'dota2split'))
