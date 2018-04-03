import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tfbldr.nodes import Linear
from tfbldr.nodes import ReLU
from tfbldr.nodes import LSTMCell
from tfbldr.nodes import VqEmbedding
from tfbldr.datasets import list_iterator
from tfbldr import get_params_dict
from tfbldr import run_loop
from tfbldr import scan
from tfbldr.datasets import make_sinewaves

import tensorflow as tf
import numpy as np
from collections import namedtuple, defaultdict

sines = make_sinewaves(50, 40, square=True)
#sines = make_sinewaves(50, 40, harmonic=True)
#sines = make_sinewaves(50, 40)
train_sines = sines[:, ::2]
train_sines = [train_sines[:, i] for i in range(train_sines.shape[1])]
valid_sines = sines[:, 1::2]
valid_sines = [valid_sines[:, i] for i in range(valid_sines.shape[1])]

"""
f, axarr = plt.subplots(4, 1)
axarr[0].plot(train_sines[0].ravel())
axarr[1].plot(valid_sines[0].ravel())
axarr[2].plot(train_sines[1].ravel())
axarr[3].plot(valid_sines[0].ravel())
plt.savefig("tmp")
"""


train_itr_random_state = np.random.RandomState(1122)
valid_itr_random_state = np.random.RandomState(12)
batch_size = 10
train_itr = list_iterator([train_sines], 10, random_state=train_itr_random_state)
valid_itr = list_iterator([valid_sines], 10, random_state=valid_itr_random_state)

random_state = np.random.RandomState(1999)

n_hid = 100
rnn_init = "truncated_normal"
forward_init = "truncated_normal"

def create_model(inp_tm1, inp_t, h1_init, c1_init, h1_q_init, c1_q_init):
    def step(x_t, h1_tm1, c1_tm1, h1_q_tm1, c1_q_tm1):
        output, s = LSTMCell([x_t, h1_q_tm1], [1, n_hid], h1_tm1, c1_tm1, n_hid,
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

        output_q_t, output_i_t, output_nst_q_t, output_emb = VqEmbedding(output, n_hid, 512,
                                                                         random_state=random_state,
                                                                         name="out_vq_emb")

        # not great
        h1_i_t = tf.cast(h1_i_t, tf.float32)
        output_i_t = tf.cast(h1_i_t, tf.float32)
        # r[0]
        rets = [output_q_t]
        # r[1:3]
        rets += [h1_t, c1_t]
        # r[3:9]
        rets += [h1_q_t, c1_q_t, h1_nst_q_t, h1_cq_t, h1_i_t, h1_emb]
        # r[9:]
        rets += [output_q_t, output_nst_q_t, output, output_i_t, output_emb]
        return rets

    outputs_info = [None,
                    h1_init, c1_init,
                    h1_q_init, c1_q_init, None, None, None, None,
                    None, None, None, None, None]
    r = scan(step, [inp_tm1], outputs_info)
    out = r[0]
    hiddens = r[1]
    cells = r[2]
    q_hiddens = r[3]
    q_cells = r[4]
    q_nst_hiddens = r[5]
    q_nvq_hiddens  = r[6]
    i_hiddens = r[7]
    emb_hiddens = r[8]
    # out == q_out
    q_out = r[9]
    q_nst_out = r[10]
    q_nvq_out = r[11]
    i_out = r[12]
    emb_out = r[13]

    l1 = Linear([out], [n_hid], n_hid, random_state=random_state, name="l1",
                init=forward_init)
    r_l1 = ReLU(l1)
    pred = Linear([r_l1], [n_hid], 1, random_state=random_state, name="out",
                  init=forward_init)
    outs_names = ["pred",
                  "hiddens",
                  "cells",
                  "q_hiddens",
                  "q_cells",
                  "q_nst_hiddens",
                  "q_nvq_hiddens",
                  "i_hiddens",
                  "emb_hiddens",
                  "q_out",
                  "q_nst_out",
                  "q_nvq_out",
                  "i_out",
                  "emb_out"]
    outs_tf = [eval(name) for name in outs_names]
    c = namedtuple("Core", outs_names)(*outs_tf)
    return c

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

        # reconstruction loss
        rec_loss = tf.reduce_mean(tf.square(r.pred - inputs_t))

        # h2h embedding losses
        alpha = 1.
        beta = 0.25
        vq_h_loss = tf.reduce_mean(tf.square(tf.stop_gradient(r.q_nvq_hiddens) - r.q_nst_hiddens))
        commit_h_loss = tf.reduce_mean(tf.square(r.q_nvq_hiddens - tf.stop_gradient(r.q_nst_hiddens)))

        # output embedding losses
        vq_o_loss = tf.reduce_mean(tf.square(tf.stop_gradient(r.q_nvq_out) - r.q_nst_out))
        commit_o_loss = tf.reduce_mean(tf.square(r.q_nvq_out - tf.stop_gradient(r.q_nst_out)))

        loss_r = rec_loss
        loss_h = alpha * vq_h_loss + beta * commit_h_loss
        loss_o = alpha * vq_o_loss + beta * commit_o_loss
        loss = loss_r + loss_h + loss_o

        params = get_params_dict()
        grads = tf.gradients(loss, params.values())
        learning_rate = 0.001
        optimizer = tf.train.AdamOptimizer(learning_rate, use_locking=True)
        assert len(grads) == len(params)
        grads = [tf.clip_by_value(g, -1., 1.) if g is not None else None for g in grads]
        j = [(g, p) for g, p in zip(grads, params.values())]
        train_step = optimizer.apply_gradients(j)

    things_names = ["inputs",
                    "inputs_tm1",
                    "inputs_t",
                    "init_hidden",
                    "init_cell",
                    "init_q_hidden",
                    "init_q_cell",
                    "loss",
                    "rec_loss",
                    "train_step"]
    things_tf = [eval(name) for name in things_names]
    things_names += r._asdict().keys()
    things_tf += r._asdict().values()
    train_model = namedtuple('Model', things_names)(*things_tf)
    for tn, tt in zip(things_names, things_tf):
        graph.add_to_collection(tn, tt)
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
             n_steps=20000,
             n_train_steps_per=5000,
             n_valid_steps_per=100)
