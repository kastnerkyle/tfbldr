import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tfbldr.nodes import Linear
from tfbldr.nodes import GRUCell
from tfbldr.datasets import list_iterator
from tfbldr import get_params_dict
from tfbldr import run_loop
from tfbldr import scan
from tfbldr.datasets import make_sinewaves

import tensorflow as tf
import numpy as np
from collections import namedtuple, defaultdict

sines = make_sinewaves(50, 40)
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

def create_model(inp_tm1, inp_t, h1_init):
    def step(x_t, h1_tm1):
        output, s = GRUCell([x_t], [1], h1_tm1, n_hid,
                            random_state=random_state,
                            name="rnn1", init=rnn_init)
        h1_t = s[0]
        return output, h1_t

    r = scan(step, [inp_tm1], [None, h1_init])
    out = r[0]
    hiddens = r[1]
    pred = Linear([out], [n_hid], 1, random_state=random_state, name="out",
                  init=forward_init)
    """
    z_e_x = create_encoder(inp, bn)
    z_q_x, z_i_x, emb = VqEmbedding(z_e_x, l_dims[-1][0], embedding_dim, random_state=random_state, name="embed")
    x_tilde = create_decoder(z_q_x, bn)
    """
    return pred, hiddens

def create_graph():
    graph = tf.Graph()
    with graph.as_default():
        inputs = tf.placeholder(tf.float32, shape=[None, batch_size, 1])
        inputs_tm1 = inputs[:-1]
        inputs_t = inputs[1:]
        init_hidden = tf.placeholder(tf.float32, shape=[batch_size, n_hid])
        pred, hiddens = create_model(inputs_tm1, inputs_t, init_hidden)
        rec_loss = tf.reduce_mean(tf.square(pred - inputs_t))
        loss = rec_loss

        params = get_params_dict()
        grads = tf.gradients(loss, params.values())

        learning_rate = 0.0002
        optimizer = tf.train.AdamOptimizer(learning_rate, use_locking=True)
        assert len(grads) == len(params)
        j = [(g, p) for g, p in zip(grads, params.values())]
        train_step = optimizer.apply_gradients(j)

    things_names = ["inputs",
                    "inputs_tm1",
                    "inputs_t",
                    "init_hidden",
                    "hiddens",
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
    init_h = np.zeros((batch_size, n_hid)).astype("float32")
    if extras["train"]:
        feed = {vs.inputs: x,
                vs.init_hidden: init_h}
        outs = [vs.rec_loss, vs.loss, vs.train_step]
        r = sess.run(outs, feed_dict=feed)
        l = r[0]
        t_l = r[1]
        step = r[2]
    else:
        feed = {vs.inputs: x,
                vs.init_hidden: init_h}
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
