from tfbldr.datasets import fetch_norvig_words
from tfbldr.datasets import list_iterator
from tfbldr.nodes import PositionalEncoding
from tfbldr.nodes import MultiheadAttention
from tfbldr.nodes import TransformerBlock
from tfbldr.nodes import CategoricalCrossEntropyLinearIndexCost
from tfbldr.nodes import LayerNorm
from tfbldr.nodes import Linear
from tfbldr.nodes import Softmax
from tfbldr.nodes import ReLU
from tfbldr import get_params_dict
from tfbldr import run_loop
from collections import namedtuple

import numpy as np
import tensorflow as tf

norvig = fetch_norvig_words()
words = norvig["data"]
maxlen = max([len(words_i) for words_i in words])
word_length_limit = 10
words = [words_i for words_i in words if len(words_i) <= word_length_limit]
#vocab = "_0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
vocab = "_0123456789abcdefghijklmnopqrstuvwxyz"
v2i = {v: k for k, v in enumerate(vocab)}
i2v = {v: k for k, v in v2i.items()}
word_inds = [np.array([v2i[wi] for wi in word_i] + [0] * (word_length_limit - len(word_i)))[..., None] for word_i in words]
rev_word_inds = [np.array([v2i[wi] for wi in word_i][::-1] + [0] * (word_length_limit - len(word_i)))[..., None] for word_i in words]

train_itr_random_state = np.random.RandomState(1122)
valid_itr_random_state = np.random.RandomState(12)

random_state = np.random.RandomState(1999)

batch_size = 64
n_syms = len(vocab)

shuffled_inds = list(range(len(word_inds)))
train_itr_random_state.shuffle(shuffled_inds)
split = 250000
train_inds = shuffled_inds[:split]
valid_inds = shuffled_inds[split:]
train_word_inds = [word_inds[i] for i in train_inds]
train_rev_word_inds = [rev_word_inds[i] for i in train_inds]

valid_word_inds = [word_inds[i] for i in valid_inds]
valid_rev_word_inds = [rev_word_inds[i] for i in valid_inds]

train_itr = list_iterator([train_word_inds, train_rev_word_inds], batch_size, random_state=train_itr_random_state)
valid_itr = list_iterator([valid_word_inds, valid_rev_word_inds], batch_size, random_state=valid_itr_random_state)

dim = 512
n_layers = 3

def create_model(inp, out):
    p_inp, emb = PositionalEncoding(inp, n_syms, dim, random_state=random_state, name="inp_pos_emb")
    prev = p_inp
    enc_atts = []
    for i in range(n_layers):
        li, atti = TransformerBlock(prev, prev, prev, dim, mask=False, random_state=random_state, name="l{}".format(i))
        prev = li
        enc_atts.append(atti)

    p_out, out_emb = PositionalEncoding(out, n_syms, dim, random_state=random_state, name="out_pos_emb")
    dec_atts = []
    prev_in = prev
    prev = p_out
    for i in range(n_layers):
        li, atti = TransformerBlock(prev_in, prev_in, prev, dim, mask=True, random_state=random_state, name="dl{}".format(i))
        prev = li
        dec_atts.append(atti)
    prob_logits = Linear([prev], [dim], n_syms, random_state=random_state, name="prob_logits")
    return prob_logits, enc_atts, dec_atts

def create_graph():
    graph = tf.Graph()
    with graph.as_default():
        inputs = tf.placeholder(tf.float32, shape=[word_length_limit, batch_size, 1])
        outputs = tf.placeholder(tf.float32, shape=[word_length_limit, batch_size, 1])
        outputs_masks = tf.placeholder(tf.float32, shape=[word_length_limit, batch_size])
        pred_logits, enc_atts, dec_atts = create_model(inputs, outputs)
        loss_i = CategoricalCrossEntropyLinearIndexCost(pred_logits, outputs)
        loss = tf.reduce_sum(outputs_masks * loss_i) / tf.reduce_sum(outputs_masks)

        params = get_params_dict()
        grads = tf.gradients(loss, params.values())

        learning_rate = 0.0002
        optimizer = tf.train.AdamOptimizer(learning_rate, use_locking=True)
        assert len(grads) == len(params)
        j = [(g, p) for g, p in zip(grads, params.values())]
        train_step = optimizer.apply_gradients(j)

    things_names = ["inputs",
                    "outputs",
                    "outputs_masks",
                    "pred_logits",
                    "enc_atts",
                    "dec_atts",
                    "loss",
                    "train_step"]
    things_tf = [eval(tn) for tn in things_names]
    assert len(things_names) == len(things_tf)
    for tn, tt in zip(things_names, things_tf):
        graph.add_to_collection(tn, tt)
    train_model = namedtuple('Model', things_names)(*things_tf)
    return graph, train_model

g, vs = create_graph()

def loop(sess, itr, extras, stateful_args):
    x, y = itr.next_batch()
    x = x.transpose(1, 0, 2)
    y = y.transpose(1, 0, 2)
    #new_y = np.zeros((y.shape[0] + 1, y.shape[1], y.shape[2]))
    #new_y[1:] = y
    y_mask = np.zeros((y.shape[0], y.shape[1], y.shape[2]))
    #y_mask[new_y > 0] = 1.
    y_mask[y > 0] = 1.
    #y_mask[0] = 1.
    y_mask = y_mask[..., 0]
    #y = new_y
    if extras["train"]:
        feed = {vs.inputs: x,
                vs.outputs: y,
                vs.outputs_masks: y_mask}
        outs = [vs.loss, vs.train_step]
        r = sess.run(outs, feed_dict=feed)
        l = r[0]
        step = r[1]
    else:
        feed = {vs.inputs: x,
                vs.outputs: y,
                vs.outputs_masks: y_mask}
        outs = [vs.loss]
        r = sess.run(outs, feed_dict=feed)
        l = r[0]
    return l, None, stateful_args

with tf.Session(graph=g) as sess:
    run_loop(sess,
             loop, train_itr,
             loop, valid_itr,
             n_steps=50000,
             n_train_steps_per=5000,
             n_valid_steps_per=5000)
