"""
Sequential Child-Combination Tree-LSTM Network for PolEval 2017 evaluation campaign
Implementation inspired by "Efficient recursive (tree-structured) neural networks in TensorFlow" available at https://github.com/erickrf/treernn
"""

import sys

import numpy as np
import tensorflow as tf

import data

CHILDREN_NB = 5

word2idx, train_data, test_data = data.parse()

embed_size = 16
label_size = 3
max_epochs = 3
lr = 0.01

with tf.variable_scope('embed'):
    embeddings = tf.get_variable('embeddings', [len(word2idx), embed_size])

with tf.variable_scope('lstm'):
    W_i = tf.get_variable('W_i', [2 * embed_size, embed_size])
    W_f = tf.get_variable('W_f', [2 * embed_size, embed_size])
    W_o = tf.get_variable('W_o', [2 * embed_size, embed_size])
    W_g = tf.get_variable('W_g', [2 * embed_size, embed_size])
    c = tf.get_variable('c', [embed_size])

with tf.variable_scope('output'):
    V = tf.get_variable('V', [embed_size, label_size])
    bs = tf.get_variable('bs', [1, label_size])

children_placeholders = [tf.placeholder(tf.int32, (None), name='children_' + str(j) + '_placeholder') for j in range(CHILDREN_NB)]
node_word_indices_placeholder = tf.placeholder(tf.int32, (None), name='node_word_indices_placeholder')
y = tf.placeholder(tf.int32, (None), name='y')


def tree_lstm():

    tensor_array = tf.TensorArray(
        tf.float32,
        size=0,
        dynamic_size=True,
        clear_after_read=False,
        infer_shape=False)

    def embed_word(word_index):
        return tf.expand_dims(tf.gather(embeddings, word_index), 0)

    def lstm(x, state):
        i = tf.nn.sigmoid(tf.matmul(tf.concat([x, state], 1), W_i))
        f = tf.nn.sigmoid(tf.matmul(tf.concat([x, state], 1), W_f))
        o = tf.nn.sigmoid(tf.matmul(tf.concat([x, state], 1), W_o))
        g = tf.nn.tanh(tf.matmul(tf.concat([x, state], 1), W_g))
        new_c = c * f + g * i
        tf.assign(c, tf.squeeze(new_c))
        r = tf.nn.tanh(new_c) * o
        return r

    def loop_body(tensor_array, i):
        node_word_index = tf.gather(node_word_indices_placeholder, i)
        children = [tf.gather(children_placeholders[j], i) for j in range(CHILDREN_NB)]
        node_tensor = embed_word(node_word_index)

        for child in children:
            node_tensor = tf.cond(child > -1,
                                  lambda: lstm(node_tensor, tensor_array.read(child)),
                                  lambda: node_tensor)
        tensor_array = tensor_array.write(i, node_tensor)
        i = tf.add(i, 1)
        return tensor_array, i

    loop_cond = lambda tensor_array, i: \
        tf.less(i, tf.squeeze(tf.shape(node_word_indices_placeholder)))
    tensor_array, _ = tf.while_loop(
        loop_cond, loop_body, [tensor_array, 0], parallel_iterations=1)

    logits = tf.matmul(tensor_array.concat(), V) + bs

    return logits


logits = tree_lstm()
prediction = tf.argmax(logits, 1)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
train_op = tf.train.AdamOptimizer(lr).minimize(loss)


def build_feed_dict(node):

    tree, labels, words, _ = node

    nodes_list = []
    data.left_traverse(-1, tree, nodes_list)
    nodes_list.pop(-1)

    feed_dict = {
        node_word_indices_placeholder: [words[node] for node in nodes_list],
        y: [labels[node] for node in nodes_list],
    }
    for j in range(CHILDREN_NB):
        feed_dict[children_placeholders[j]] = [tree[node][j] if len(tree[node]) > j else -1 for node in nodes_list]
    for j in range(CHILDREN_NB):
        feed_dict[children_placeholders[j]] = [nodes_list.index(f) if f > -1 else -1 for f in feed_dict[children_placeholders[j]]]

    return feed_dict


def predict(sess, data):

    y_pred = []
    y_true = []

    for step, t in enumerate(data):
        feed_dict = build_feed_dict(t)
        pred, y_i = sess.run([prediction, y], feed_dict=feed_dict)
        y_pred.append(pred)
        y_true.append(y_i)

    list_flatten = lambda l: [item for sublist in l for item in sublist]

    y_pred = list_flatten(y_pred)
    y_true = list_flatten(y_true)

    acc = np.equal(y_true, y_pred).mean()

    return acc


with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())

    for epoch in range(1, max_epochs + 1):
        print('\n\nepoch {}'.format(epoch))

        for step, t in enumerate(train_data):
            feed_dict = build_feed_dict(t)
            sess.run([train_op], feed_dict=feed_dict)
            sys.stdout.write('\r{} / {}'.format(step, len(train_data)))
            sys.stdout.flush()

        acc = predict(sess, train_data)
        sys.stdout.write('\r{} / {}\ttrain acc: {}'.format(step, len(train_data), acc))
        sys.stdout.flush()

    saver.save(sess, 'models/model')

    saver.restore(sess, 'models/model')
    acc = predict(sess, test_data)
    sys.stdout.write('\n\ntest acc: {}'.format(acc))# test
