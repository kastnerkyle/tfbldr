from tfbldr.nodes import Linear
from tfbldr.nodes import LSTMCell
from tfbldr.nodes import VqEmbedding
from tfbldr.nodes import Softmax
from tfbldr.nodes import OneHot
from tfbldr.nodes import CategoricalCrossEntropyIndexCost
from tfbldr.datasets import char_textfile_iterator
from tfbldr import get_params_dict
from tfbldr import run_loop
from tfbldr import scan

import numpy as np
import tensorflow as tf
from collections import namedtuple


batch_size = 20
seq_length = 50
train_random_state = np.random.RandomState(182)
valid_random_state = np.random.RandomState(7)
train_itr = char_textfile_iterator("ptb_data/ptb.train.txt", batch_size, seq_length,
                                   random_state=train_random_state)
valid_itr = char_textfile_iterator("ptb_data/ptb.valid.txt", batch_size, seq_length,
                                   random_state=valid_random_state)

random_state = np.random.RandomState(1177)

n_hid = 512
n_emb = 512
n_inputs = len(train_itr.char2ind)
rnn_init = "truncated_normal"
forward_init = "truncated_normal"

def create_model(inp_tm1, inp_t, h1_init, c1_init, h1_q_init, c1_q_init):
    oh_tm1 = OneHot(inp_tm1, n_inputs)
    p_tm1 = Linear([oh_tm1], [n_inputs], n_hid, random_state=random_state, name="proj",
                  init=forward_init)
    def step(x_t, h1_tm1, c1_tm1, h1_q_tm1, c1_q_tm1):
        output, s = LSTMCell([x_t], [n_hid], h1_tm1, c1_tm1, n_hid,
                             random_state=random_state,
                             name="rnn1", init=rnn_init)
        h1_t = s[0]
        c1_t = s[1]

        output, s = LSTMCell([h1_t], [n_hid], h1_q_tm1, c1_q_tm1, n_hid,
                             random_state=random_state,
                             name="rnn1_q", init=rnn_init)
        h1_cq_t = s[0]
        c1_q_t = s[1]

        h1_q_t, h1_i_t, h1_nst_q_t, h1_emb = VqEmbedding(h1_cq_t, n_hid, n_emb,
                                                         random_state=random_state,
                                                         name="h1_vq_emb")

        # not great
        h1_i_t = tf.cast(h1_i_t, tf.float32)
        return output, h1_t, c1_t, h1_q_t, c1_q_t, h1_nst_q_t, h1_cq_t, h1_i_t

    r = scan(step, [p_tm1], [None, h1_init, c1_init, h1_q_init, c1_q_init, None, None, None])
    out = r[0]
    hiddens = r[1]
    cells = r[2]
    q_hiddens = r[3]
    q_cells = r[4]
    q_nst_hiddens = r[5]
    q_nvq_hiddens  = r[6]
    i_hiddens = r[7]

    pred = Linear([out], [n_hid], n_inputs, random_state=random_state, name="out",
                  init=forward_init)
    pred_sm = Softmax(pred)
    return pred_sm, pred, hiddens, cells, q_hiddens, q_cells, q_nst_hiddens, q_nvq_hiddens, i_hiddens

def create_graph():
    graph = tf.Graph()
    with graph.as_default():
        inputs = tf.placeholder(tf.float32, shape=[None, batch_size, 1])
        inputs_tm1 = inputs[:-1]
        inputs_t = inputs[1:]
        init_hidden = tf.placeholder(tf.float32, shape=[batch_size, n_hid])
        init_cell = tf.placeholder(tf.float32, shape=[batch_size, n_hid])
        init_q_hidden = tf.placeholder(tf.float32, shape=[batch_size, n_hid])
        init_q_cell = tf.placeholder(tf.float32, shape=[batch_size, n_hid])
        r = create_model(inputs_tm1, inputs_t, init_hidden, init_cell, init_q_hidden, init_q_cell)
        pred_sm, pred, hiddens, cells, q_hiddens, q_cells, q_nst_hiddens, q_nvq_hiddens, i_hiddens = r
        rec_loss = tf.reduce_mean(CategoricalCrossEntropyIndexCost(pred_sm, inputs_t))

        alpha = 1.
        beta = 0.25
        vq_h_loss = tf.reduce_mean(tf.square(tf.stop_gradient(q_nvq_hiddens) - q_nst_hiddens))
        commit_h_loss = tf.reduce_mean(tf.square(q_nvq_hiddens - tf.stop_gradient(q_nst_hiddens)))

        loss = rec_loss + alpha * vq_h_loss + beta * commit_h_loss

        params = get_params_dict()
        grads = tf.gradients(loss, params.values())
        learning_rate = 0.0001
        optimizer = tf.train.AdamOptimizer(learning_rate, use_locking=True)
        assert len(grads) == len(params)
        grads = [tf.clip_by_value(g, -10., 10.) if g is not None else None for g in grads]
        j = [(g, p) for g, p in zip(grads, params.values())]
        train_step = optimizer.apply_gradients(j)

    things_names = ["inputs",
                    "inputs_tm1",
                    "inputs_t",
                    "init_hidden",
                    "init_cell",
                    "init_q_hidden",
                    "init_q_cell",
                    "hiddens",
                    "cells",
                    "q_hiddens",
                    "q_cells",
                    "q_nvq_hiddens",
                    "i_hiddens",
                    "pred_sm",
                    "pred",
                    "loss",
                    "rec_loss",
                    "train_step"]
    things_tf = [inputs,
                 inputs_tm1,
                 inputs_t,
                 init_hidden,
                 init_cell,
                 init_q_hidden,
                 init_q_cell,
                 hiddens,
                 cells,
                 q_hiddens,
                 q_cells,
                 q_nvq_hiddens,
                 i_hiddens,
                 pred_sm,
                 pred,
                 loss,
                 rec_loss,
                 train_step]
    assert len(things_names) == len(things_tf)
    for tn, tt in zip(things_names, things_tf):
        graph.add_to_collection(tn, tt)
    train_model = namedtuple('Model', things_names)(*things_tf)
    return graph, train_model

g, vs = create_graph()

def loop(sess, itr, extras, stateful_args):
    x, reset = itr.next_batch()
    init_h = np.zeros((batch_size, n_hid)).astype("float32")
    init_c = np.zeros((batch_size, n_hid)).astype("float32")
    init_q_h = np.zeros((batch_size, n_hid)).astype("float32")
    init_q_c = np.zeros((batch_size, n_hid)).astype("float32")
    if extras["train"]:
        feed = {vs.inputs: x,
                vs.init_hidden: init_h,
                vs.init_cell: init_c,
                vs.init_q_hidden: init_q_h,
                vs.init_q_cell: init_q_c}
        outs = [vs.rec_loss, vs.loss, vs.train_step]
        r = sess.run(outs, feed_dict=feed)
        l = r[0]
        t_l = r[1]
        step = r[2]
    else:
        feed = {vs.inputs: x,
                vs.init_hidden: init_h,
                vs.init_cell: init_c,
                vs.init_q_hidden: init_q_h,
                vs.init_q_cell: init_q_c}
        outs = [vs.rec_loss]
        r = sess.run(outs, feed_dict=feed)
        l = r[0]
    return l, None, stateful_args

with tf.Session(graph=g) as sess:
    run_loop(sess,
             loop, train_itr,
             loop, valid_itr,
             n_steps=200000,
             n_train_steps_per=10000,
             n_valid_steps_per=500)
