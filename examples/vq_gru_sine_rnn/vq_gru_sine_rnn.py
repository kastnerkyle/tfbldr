import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tfbldr.nodes import Linear
from tfbldr.nodes import GRUCell
from tfbldr.nodes import ReLU
from tfbldr.nodes import VqEmbedding
from tfbldr.datasets import list_iterator
from tfbldr import get_params_dict
from tfbldr import run_loop
from tfbldr import scan
from tfbldr.datasets import make_sinewaves

import tensorflow as tf
import numpy as np
from collections import namedtuple, defaultdict

sines = make_sinewaves(50, 40, harmonic=True)
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

n_hid = 128
n_emb = 512
n_split = 8
assert n_hid // n_split == n_hid / (float(n_split))

rnn_init = "truncated_normal"
forward_init = "truncated_normal"

def create_model(inp_tm1, h1_q_init):
    def step(x_t, h1_tm1):
        output, s = GRUCell([x_t], [1], h1_tm1, n_hid,
                            random_state=random_state,
                            name="rnn1", init=rnn_init)
        h1_cq_t = s[0]

        """
        output, s = LSTMCell([h1_t], [n_hid], h1_q_tm1, c1_q_tm1, n_hid,
                             random_state=random_state,
                             name="rnn1_q", init=rnn_init)
        h1_cq_t = s[0]
        c1_q_t = s[1]
        """
        qhs = []
        ihs = []
        nst_qhs = []
        embs = []
        for i in list(range(n_split)):
            e_div = int(n_hid / n_split)
            h1_q_t, h1_i_t, h1_nst_q_t, h1_emb = VqEmbedding(h1_cq_t[:, i * e_div:(i + 1) * e_div],
                                                             e_div,
                                                             n_emb,
                                                             random_state=random_state,
                                                             # shared space?
                                                             name="h1_vq_emb")
                                                             #name="h1_{}_vq_emb".format(i))
            qhs.append(h1_q_t)
            ihs.append(h1_i_t[:, None])
            nst_qhs.append(h1_nst_q_t)
            embs.append(h1_emb)
        h1_q_t = tf.concat(qhs, axis=-1)
        h1_nst_q_t = tf.concat(nst_qhs, axis=-1)
        h1_i_t = tf.concat(ihs, axis=-1)

        # not great
        h1_i_t = tf.cast(h1_i_t, tf.float32)
        return output, h1_q_t, h1_nst_q_t, h1_cq_t, h1_i_t

    r = scan(step, [inp_tm1], [None, h1_q_init, None, None, None])
    out = r[0]
    q_hiddens = r[1]
    q_nst_hiddens = r[2]
    q_nvq_hiddens  = r[3]
    i_hiddens = r[4]

    l1 = Linear([out], [n_hid], n_hid, random_state=random_state, name="l1",
                 init=forward_init)
    r_l1 = ReLU(l1)
    pred = Linear([r_l1], [n_hid], 1, random_state=random_state, name="out",
                  init=forward_init)

    return pred, q_hiddens, q_nst_hiddens, q_nvq_hiddens, i_hiddens

def create_graph():
    graph = tf.Graph()
    with graph.as_default():
        inputs = tf.placeholder(tf.float32, shape=[None, batch_size, 1])
        inputs_tm1 = inputs[:-1]
        inputs_t = inputs[1:]
        init_q_hidden = tf.placeholder(tf.float32, shape=[batch_size, n_hid])
        r = create_model(inputs_tm1, init_q_hidden)
        pred, q_hiddens, q_nst_hiddens, q_nvq_hiddens, i_hiddens = r
        rec_loss = tf.reduce_mean(tf.square(pred - inputs_t))

        alpha = 1.
        beta = 0.25
        vq_h_loss = tf.reduce_mean(tf.square(tf.stop_gradient(q_nvq_hiddens) - q_nst_hiddens))
        commit_h_loss = tf.reduce_mean(tf.square(q_nvq_hiddens - tf.stop_gradient(q_nst_hiddens)))

        loss = rec_loss + alpha * vq_h_loss + beta * commit_h_loss

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
                    "init_q_hidden",
                    "q_hiddens",
                    "q_nvq_hiddens",
                    "i_hiddens",
                    "pred",
                    "loss",
                    "rec_loss",
                    "train_step"]
    things_tf = [eval(name) for name in things_names]
    for tn, tt in zip(things_names, things_tf):
        graph.add_to_collection(tn, tt)
    train_model = namedtuple('Model', things_names)(*things_tf)
    return graph, train_model

g, vs = create_graph()

def loop(sess, itr, extras, stateful_args):
    x, = itr.next_batch()
    x = x.transpose(1, 0, 2)
    init_q_h = np.zeros((batch_size, n_hid)).astype("float32")
    if extras["train"]:
        feed = {vs.inputs: x,
                vs.init_q_hidden: init_q_h}
        outs = [vs.rec_loss, vs.loss, vs.train_step]
        r = sess.run(outs, feed_dict=feed)
        l = r[0]
        t_l = r[1]
        step = r[2]
    else:
        feed = {vs.inputs: x,
                vs.init_q_hidden: init_q_h}
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
