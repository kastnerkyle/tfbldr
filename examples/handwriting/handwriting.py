from __future__ import print_function
from extras import fetch_iamondb, rsync_fetch, list_iterator
from tfdllib import Linear, scan, get_params_dict, print_network
from tfdllib import GRUFork
from tfdllib import GRU
from tfdllib import GaussianAttention
from tfdllib import LogitBernoulliAndCorrelatedLogitGMM
from tfdllib import LogitBernoulliAndCorrelatedLogitGMMCost
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
n_mdn = 20
h_dim = 400
cut_len = 300
n_batch = trace_mb.shape[1]

X_char = tf.placeholder(tf.float32, shape=[None, n_batch, n_letters],
                        name="X_char")
X_char_mask = tf.placeholder(tf.float32, shape=[None, n_batch],
                             name="X_char_mask")
y_pen = tf.placeholder(tf.float32, shape=[None, n_batch, 3],
                       name="y_pen")
y_pen_mask = tf.placeholder(tf.float32, shape=[None, n_batch],
                            name="y_pen_mask")
init_h1 = tf.placeholder(tf.float32, [n_batch, h_dim],
                         name="init_h1")
init_h2 = tf.placeholder(tf.float32, [n_batch, h_dim],
                         name="init_h2")
init_h3 = tf.placeholder(tf.float32, [n_batch, h_dim],
                         name="init_h3")
init_att_h = tf.placeholder(tf.float32, [n_batch, h_dim],
                            name="init_att_h")
init_att_k = tf.placeholder(tf.float32, [n_batch, n_attention],
                            name="init_att_k")
init_att_w = tf.placeholder(tf.float32, [n_batch, n_letters],
                            name="init_att_w")
bias = tf.placeholder_with_default(tf.zeros(shape=[]), shape=[],
                                   name="bias")

y_tm1 = y_pen[:-1, :, :]
y_t = y_pen[1:, :, :]
y_mask_tm1 = y_pen_mask[:-1]

rnn_init = "normal"
forward_init = "normal"
use_weight_norm = False
norm_clip = True
if norm_clip:
    grad_clip = 10.
else:
    grad_clip = 100.

proj_inp = Linear([y_tm1], [3], h_dim, random_state=random_state,
                  weight_norm=use_weight_norm, init=forward_init,
                  name="proj")

def step(inp_t, inp_mask_t, h1_tm1, h2_tm1, h3_tm1, att_h_tm1, att_k_tm1, att_w_tm1):
    r = GaussianAttention([inp_t], [h_dim],
                          att_h_tm1,
                          att_k_tm1,
                          att_w_tm1,
                          X_char,
                          n_letters,
                          h_dim,
                          step_mask=inp_mask_t,
                          conditioning_mask=X_char_mask,
                          weight_norm=use_weight_norm,
                          random_state=random_state,
                          name="cond_text",
                          init=rnn_init,
                          )

    att_h_t, att_k_t, att_w_t, att_phi_t = r

    fork1_t, fork1_gate_t = GRUFork([att_w_t], [n_letters], h_dim,
                                    weight_norm=use_weight_norm,
                                    random_state=random_state,
                                    name="gru1_fork",
                                    init=forward_init)
    h1_t = GRU(fork1_t, fork1_gate_t, h1_tm1, h_dim, h_dim,
               mask=inp_mask_t,
               weight_norm=use_weight_norm,
               random_state=random_state, init=rnn_init, name="gru1")

    fork2_t, fork2_gate_t = GRUFork([h1_t], [h_dim], h_dim,
                                    weight_norm=use_weight_norm,
                                    random_state=random_state,
                                    name="gru2_fork",
                                    init=forward_init)
    h2_t = GRU(fork2_t, fork2_gate_t, h2_tm1, h_dim, h_dim,
               mask=inp_mask_t,
               weight_norm=use_weight_norm,
               random_state=random_state, init=rnn_init, name="gru2")

    fork3_t, fork3_gate_t = GRUFork([h2_t], [h_dim], h_dim,
                                    weight_norm=use_weight_norm,
                                    random_state=random_state,
                                    name="gru3_fork",
                                    init=forward_init)
    h3_t = GRU(fork3_t, fork3_gate_t, h3_tm1, h_dim, h_dim,
               mask=inp_mask_t,
               weight_norm=use_weight_norm,
               random_state=random_state, init=rnn_init, name="gru3")
    return h1_t, h2_t, h3_t, att_h_t, att_k_t, att_w_t

o = scan(step,
        [proj_inp, y_mask_tm1],
        [init_h1, init_h2, init_h3, init_att_h, init_att_k, init_att_w])
h1_o = o[0]
h2_o = o[1]
h3_o = o[2]
att_h = o[3]
att_k = o[4]
att_w = o[5]

h1_o = tf.identity(h1_o, name="h1_o")
h2_o = tf.identity(h2_o, name="h2_o")
h3_o = tf.identity(h3_o, name="h3_o")
att_h = tf.identity(att_h, name="att_h")
att_k = tf.identity(att_k, name="att_k")
att_w = tf.identity(att_w, name="att_w")

p = LogitBernoulliAndCorrelatedLogitGMM([h3_o], [h_dim], name="b_gmm",
                                        n_components=n_mdn,
                                        weight_norm=use_weight_norm,
                                        random_state=random_state,
                                        init=forward_init)
logit_bernoullis = p[0]
logit_coeffs = p[1]
mus = p[2]
logit_sigmas = p[3]
corrs = p[4]

#cost = BernoulliAndCorrelatedGMMCost(
#    bernoullis, coeffs, mus, sigmas, corrs, y_t, name="cost")
cost = LogitBernoulliAndCorrelatedLogitGMMCost(
    logit_bernoullis, logit_coeffs, mus, logit_sigmas, corrs, y_t, name="cost")
loss = tf.reduce_mean(cost)
#masked_cost = y_pen_mask * cost
#loss = tf.reduce_sum(cost / (tf.reduce_sum(y_pen_mask) + 1.))

params_dict = get_params_dict()
params = params_dict.values()
grads = tf.gradients(loss, params)

if norm_clip:
    grads, _ = tf.clip_by_global_norm(grads, grad_clip)
else:
    grads = [tf.clip_by_value(grad, -grad_clip, grad_clip) for grad in grads]

learning_rate = 0.0002
"""
steps = tf.Variable(0.)
learning_rate = tf.train.exponential_decay(0.001, steps, staircase=True, decay_steps=10000,
        decay_rate=0.5)
"""

opt = tf.train.AdamOptimizer(learning_rate=learning_rate, use_locking=True)
#updates = opt.apply_gradients(zip(grads, params), global_step=steps)
updates = opt.apply_gradients(zip(grads, params))

summary = tf.summary.merge([
    tf.summary.scalar('loss', loss),
])


def make_tbptt_batches(list_of_arrays, seq_len, drop_if_less_than=10):
    """ Assume axis 0 is sequence axis """
    core_shp = list_of_arrays[0].shape
    split_arrays = []
    for la in list_of_arrays:
        assert la.shape[0] == core_shp[0]
        last = 0
        split_arrays.append([])
        while True:
            this = last + seq_len
            if len(la) > this:
                split_arrays[-1].append(la[last:this])
                last = this
            else:
                split_arrays[-1].append(la[last:])
                break

    core_len = len(split_arrays[0])
    for sa in split_arrays:
        assert len(sa) == core_len

    # now paired and grouped
    new_split_arrays = zip(*iter(split_arrays))
    new_split_arrays = [nsa for nsa in new_split_arrays if len(nsa[0]) >= drop_if_less_than]
    return new_split_arrays


def loop(itr, sess, extras):
    trace_mb, trace_mask, text_mb, text_mask = next(itr)
    init_h1_np = np.zeros((n_batch, h_dim)).astype("float32")
    init_h2_np = np.zeros((n_batch, h_dim)).astype("float32")
    init_h3_np = np.zeros((n_batch, h_dim)).astype("float32")
    init_att_h_np = np.zeros((n_batch, h_dim)).astype("float32")
    init_att_k_np = np.zeros((n_batch, n_attention)).astype("float32")
    init_att_w_np = np.zeros((n_batch, n_letters)).astype("float32")

    tbptt_batches = make_tbptt_batches([trace_mb, trace_mask], cut_len)
    overall_train_loss = 0.
    overall_train_len = 0.
    train_summaries = []
    for batch in tbptt_batches:
        tbptt_trace_mb = batch[0]
        tbptt_trace_mask = batch[1]
        feed = {X_char: text_mb,
                X_char_mask: text_mask,
                y_pen: tbptt_trace_mb,
                y_pen_mask: tbptt_trace_mask,
                init_h1: init_h1_np,
                init_h2: init_h2_np,
                init_h3: init_h3_np,
                init_att_h: init_att_h_np,
                init_att_k: init_att_k_np,
                init_att_w: init_att_w_np}

        if extras["train"]:
            outs = [loss, summary, updates, h1_o, h2_o, h3_o, att_h, att_k, att_w]
            p = sess.run(outs, feed)
            train_loss = p[0]
            train_summary = p[1]
            _ = p[2]
            h1_np = p[3]
            h2_np = p[4]
            h3_np = p[5]
            att_h_np = p[6]
            att_k_np = p[7]
            att_w_np = p[8]

            init_h1_np = h1_np[-1]
            init_h2_np = h2_np[-1]
            init_h3_np = h3_np[-1]
            init_att_h_np = att_h_np[-1]
            init_att_k_np = att_k_np[-1]
            init_att_w_np = att_w_np[-1]
            # undo the mean, so we get the correct aggregate
            overall_train_loss += len(att_k_np) * train_loss
            overall_train_len += len(att_k_np)
            train_summaries.append(train_summary)
            #train_loss, train_summary, _ = sess.run(outs, feed)
        else:
            print("VALID NOT YET CODED")
            outs = [loss, summary]
            train_loss, train_summary = sess.run(outs, feed)[0]
    final_train_loss = overall_train_loss / overall_train_len
    return [final_train_loss, train_summaries]

sess = tf.Session()

with sess.as_default():
    tf.global_variables_initializer().run()
    print_network(get_params_dict())
    av = tf.global_variables()
    model_saver = tf.train.Saver(max_to_keep=2)
    experiment_path = next_experiment_path()
    print("Using experiment path {}".format(experiment_path))

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
                l = ret[0]
                print('\r[{:5d}/{:5d}] loss = {}'.format(b * n_batch, len(target), l), end='')
                s = ret[1]
                for si in s:
                    summary_writer.add_summary(si, global_step=global_step)
                    global_step += 1
        except StopIteration:
            print(" ")
            print("Training done")
            model_saver.save(sess, os.path.join(experiment_path, 'models', 'model'), global_step=e)
            train_itr.reset()
