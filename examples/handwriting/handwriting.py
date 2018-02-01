from __future__ import print_function
from extras import fetch_iamondb, rsync_fetch, list_iterator
from tfdllib import Linear, scan, get_params_dict, print_network
from tfdllib import GRUFork
from tfdllib import GRU
from tfdllib import GaussianAttention
from tfdllib import LogBernoulliAndCorrelatedLogGMM
from tfdllib import LogBernoulliAndCorrelatedLogGMMCost
import tensorflow as tf
import numpy as np

import os
import shutil

# https://github.com/Grzego/handwriting-generation/blob/master/utils.py
def next_experiment_path():
    """
    creates paths for new experiment
    returns path for next experiment
    """

    idx = 0
    path = os.path.join('summary', 'experiment-{}')
    while os.path.exists(path.format(idx)):
        idx += 1
    path = path.format(idx)
    os.makedirs(os.path.join(path, 'models'))
    os.makedirs(os.path.join(path, 'backup'))
    for file in filter(lambda x: x.endswith('.py'), os.listdir('.')):
        shutil.copy2(file, os.path.join(path, 'backup'))
    return path

iamondb = rsync_fetch(fetch_iamondb, "leto01")
data = iamondb["data"]
target = iamondb["target"]

X = [xx.astype("float32") for xx in data]
y = [yy.astype("float32") for yy in target]

train_itr = list_iterator([X, y], minibatch_size=50, axis=1,
                          stop_index=10000,
                          make_mask=True)
trace_mb, trace_mask, text_mb, text_mask = next(train_itr)
train_itr.reset()
n_letters = len(iamondb["vocabulary"])
random_state = np.random.RandomState(1999)
n_attention = 10
h_dim = 256
n_batch = trace_mb.shape[1]

X_char = tf.placeholder(tf.float32, shape=[None, n_batch, n_letters])
y_pen = tf.placeholder(tf.float32, shape=[None, n_batch, 3])
init_h1 = tf.placeholder(tf.float32, [n_batch, h_dim])
init_h2 = tf.placeholder(tf.float32, [n_batch, h_dim])
init_h3 = tf.placeholder(tf.float32, [n_batch, h_dim])
init_att_h = tf.placeholder(tf.float32, [n_batch, h_dim])
init_att_k = tf.placeholder(tf.float32, [n_batch, n_attention])
init_att_w = tf.placeholder(tf.float32, [n_batch, n_letters])
bias = tf.placeholder_with_default(tf.zeros(shape=[]), shape=[])

y_tm1 = y_pen[:-1, :, :]
y_t = y_pen[1:, :, :]

rnn_init = "truncated_normal"
forward_init = "truncated_normal"
use_weight_norm = False
norm_clip = True
if norm_clip:
    grad_clip = 3.
else:
    grad_clip = 100.

proj_inp = Linear([y_tm1], [3], h_dim, random_state=random_state,
                  weight_norm=use_weight_norm, init=forward_init,
                  name="proj")

def step(inp_t, h1_tm1, h2_tm1, h3_tm1, att_h_tm1, att_k_tm1, att_w_tm1):
    r = GaussianAttention([inp_t], [h_dim],
                          att_h_tm1,
                          att_k_tm1,
                          att_w_tm1,
                          X_char,
                          n_letters,
                          h_dim,
                          weight_norm=use_weight_norm,
                          random_state=random_state,
                          name="cond_text",
                          init=rnn_init,
                          )

    att_h_t, att_k_t, att_w_t = r

    fork1_t, fork1_gate_t = GRUFork([inp_t, att_w_t], [h_dim, n_letters], h_dim,
                                    weight_norm=use_weight_norm,
                                    random_state=random_state,
                                    name="gru1_fork",
                                    init=forward_init)
    h1_t = GRU(fork1_t, fork1_gate_t, h1_tm1, h_dim, h_dim,
               weight_norm=use_weight_norm,
               random_state=random_state, init=rnn_init, name="gru1")

    fork2_t, fork2_gate_t = GRUFork([h1_t, att_w_t], [h_dim, n_letters], h_dim,
                                    weight_norm=use_weight_norm,
                                    random_state=random_state,
                                    name="gru2_fork",
                                    init=forward_init)
    h2_t = GRU(fork2_t, fork2_gate_t, h2_tm1, h_dim, h_dim,
               weight_norm=use_weight_norm,
               random_state=random_state, init=rnn_init, name="gru2")

    fork3_t, fork3_gate_t = GRUFork([h2_t, att_w_t], [h_dim, n_letters], h_dim,
                                    weight_norm=use_weight_norm,
                                    random_state=random_state,
                                    name="gru3_fork",
                                    init=forward_init)
    h3_t = GRU(fork3_t, fork3_gate_t, h3_tm1, h_dim, h_dim,
               weight_norm=use_weight_norm,
               random_state=random_state, init=rnn_init, name="gru3")
    return h1_t, h2_t, h3_t, att_h_t, att_k_t, att_w_t


o = scan(step, [proj_inp], [init_h1, init_h2, init_h3,
                            init_att_h, init_att_k, init_att_w])
h1_o = o[0]
att_h = o[1]
att_k = o[2]
att_w = o[3]

p = LogBernoulliAndCorrelatedLogGMM([h1_o], [h_dim], name="b_log_gmm",
                                    weight_norm=use_weight_norm,
                                    random_state=random_state,
                                    init=forward_init)
log_bernoullis = p[0]
coeffs = p[1]
mus = p[2]
log_sigmas = p[3]
corrs = p[4]

cost = LogBernoulliAndCorrelatedLogGMMCost(
    log_bernoullis, coeffs, mus, log_sigmas, corrs, y_t)
loss = tf.reduce_mean(cost)
summary = tf.summary.merge([
    tf.summary.scalar('loss', loss)
])

params_dict = get_params_dict()
params = params_dict.values()
grads = tf.gradients(cost, params)

learning_rate = .0002
if norm_clip:
    grads, _ = tf.clip_by_global_norm(grads, grad_clip)
else:
    grads = [tf.clip_by_value(grad, -grad_clip, grad_clip) for grad in grads]

opt = tf.train.AdamOptimizer(learning_rate, use_locking=True)
updates = opt.apply_gradients(zip(grads, params))

def loop(itr, sess, extras):
    trace_mb, trace_mask, text_mb, text_mask = next(itr)
    init_h1_np = np.zeros((n_batch, h_dim)).astype("float32")
    init_h2_np = np.zeros((n_batch, h_dim)).astype("float32")
    init_h3_np = np.zeros((n_batch, h_dim)).astype("float32")
    init_att_h_np = np.zeros((n_batch, h_dim)).astype("float32")
    init_att_k_np = np.zeros((n_batch, n_attention)).astype("float32")
    init_att_w_np = np.zeros((n_batch, n_letters)).astype("float32")
    feed = {X_char: text_mb,
            y_pen: trace_mb,
            init_h1: init_h1_np,
            init_h2: init_h2_np,
            init_h3: init_h3_np,
            init_att_h: init_att_h_np,
            init_att_k: init_att_k_np,
            init_att_w: init_att_w_np}

    if extras["train"]:
        outs = [loss, summary, updates]
        train_loss, train_summary, _ = sess.run(outs, feed)
    else:
        outs = [loss, summary]
        train_loss, train_summary = sess.run(outs, feed)[0]
    return [train_loss, train_summary]

sess = tf.Session()

with sess.as_default():
    tf.global_variables_initializer().run()
    print_network(get_params_dict())
    av = tf.global_variables()
    model_saver = tf.train.Saver(max_to_keep=2)
    experiment_path = next_experiment_path()

    global_step = 0
    summary_writer = tf.summary.FileWriter(experiment_path, flush_secs=10)
    summary_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START),
                                   global_step=global_step)

    for e in range(30):
        print(" ")
        print("Epoch {}".format(e))
        try:
            print(" ")
            print("Training started")
            b = 0
            while True:
                ret = loop(train_itr, sess, {"train": True})
                b += 1
                global_step += 1
                l = ret[0]
                s = ret[1]
                print('\r[{:5d}/{:5d}] loss = {}'.format(b * n_batch, len(target), l), end='')
                summary_writer.add_summary(s, global_step=global_step)
        except StopIteration:
            print(" ")
            print("Training done")
            model_saver.save(sess, os.path.join(experiment_path, 'models', 'model'), global_step=e)
            train_itr.reset()
