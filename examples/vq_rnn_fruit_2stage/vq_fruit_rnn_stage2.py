import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tfbldr.nodes import Linear
from tfbldr.nodes import LSTMCell
from tfbldr.nodes import VqEmbedding
from tfbldr.nodes import Softmax
from tfbldr.nodes import OneHot
from tfbldr.nodes import CategoricalCrossEntropyIndexCost
from tfbldr.datasets import list_iterator
from tfbldr import get_params_dict
from tfbldr import run_loop
from tfbldr import scan
from tfbldr.datasets import make_sinewaves

import tensorflow as tf
import numpy as np
from collections import namedtuple, defaultdict


train_d = np.load("stage1_data/train_data.npz")
valid_d = np.load("stage1_data/valid_data.npz")

"""
f, axarr = plt.subplots(4, 1)
axarr[0].plot(train_sines[0].ravel())
axarr[1].plot(valid_sines[0].ravel())
axarr[2].plot(train_sines[1].ravel())
axarr[3].plot(valid_sines[0].ravel())
plt.savefig("tmp")
"""

# 512 states
train_arr = train_d["z_i_x"]
valid_arr = valid_d["z_i_x"]
train_data = [train_arr[i].transpose(1, 0) for i in range(len(train_arr))]
valid_data = [valid_arr[i].transpose(1, 0) for i in range(len(valid_arr))]

train_itr_random_state = np.random.RandomState(1122)
valid_itr_random_state = np.random.RandomState(12)
batch_size = 10
train_itr = list_iterator([train_data], 10, random_state=train_itr_random_state)
valid_itr = list_iterator([valid_data], 10, random_state=valid_itr_random_state)

random_state = np.random.RandomState(1999)

n_hid = 100
rnn_init = "truncated_normal"
forward_init = "truncated_normal"

def create_model(inp_tm1, inp_t, h1_init, c1_init, h1_q_init, c1_q_init):
    oh_tm1 = OneHot(inp_tm1, 512)
    p_tm1 = Linear([oh_tm1], [512], n_hid, random_state=random_state, name="proj",
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

        h1_q_t, h1_i_t, h1_nst_q_t, h1_emb = VqEmbedding(h1_cq_t, n_hid, 512,
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

    pred = Linear([out], [n_hid], 512, random_state=random_state, name="out",
                  init=forward_init)
    pred_sm = Softmax(pred)
    return pred_sm, pred, hiddens, cells, q_hiddens, q_cells, q_nst_hiddens, q_nvq_hiddens, i_hiddens, oh_tm1

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
        pred_sm, pred, hiddens, cells, q_hiddens, q_cells, q_nst_hiddens, q_nvq_hiddens, i_hiddens, oh_tm1 = r
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
                    "pred",
                    "pred_sm",
                    "oh_tm1",
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
                 pred,
                 pred_sm,
                 oh_tm1,
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
    x, = itr.next_batch()
    x = x.transpose(1, 0, 2)
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
        #outs = [vs.rec_loss, vs.loss, vs.train_step, vs.oh_tm1, vs.pred_sm]
        outs = [vs.rec_loss, vs.loss, vs.train_step]
        r = sess.run(outs, feed_dict=feed)
        l = r[0]
        t_l = r[1]
        step = r[2]
        #oh = r[3]
        #pred_sm = r[4]
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
             n_steps=50000,
             n_train_steps_per=5000,
             n_valid_steps_per=100)
