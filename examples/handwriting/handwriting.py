from __future__ import print_function
from extras import fetch_iamondb, rsync_fetch
from batch_generator import BatchGenerator

import tensorflow as tf
import numpy as np

import os
import shutil

from tfdllib import scan
from tfdllib import get_params_dict
from tfdllib import print_network
from tfdllib import GaussianAttentionCell
from tfdllib import LSTMCell
from tfdllib import LogitBernoulliAndCorrelatedLogitGMM
from tfdllib import LogitBernoulliAndCorrelatedLogitGMMCost

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

n_letters = len(iamondb["vocabulary"])
random_state = np.random.RandomState(1999)
iteration_seed = 42
n_epochs = 30
n_attention = 10
n_mdn = 20
h_dim = 400
cut_len = 256
n_batch = 64

X_char = tf.placeholder(tf.float32, shape=[None, n_batch, n_letters],
                        name="X_char")
y_pen_tm1 = tf.placeholder(tf.float32, shape=[None, n_batch, 3],
                       name="y_pen_tm1")
y_pen_t = tf.placeholder(tf.float32, shape=[None, n_batch, 3],
                       name="y_pen_t")
init_att_h = tf.placeholder(tf.float32, [n_batch, h_dim],
                            name="init_att_h")
init_att_c = tf.placeholder(tf.float32, [n_batch, h_dim],
                            name="init_att_c")
init_att_k = tf.placeholder(tf.float32, [n_batch, n_attention],
                            name="init_att_k")
init_att_w = tf.placeholder(tf.float32, [n_batch, n_letters],
                            name="init_att_w")

init_h1 = tf.placeholder(tf.float32, [n_batch, h_dim],
                         name="init_h1")
init_c1 = tf.placeholder(tf.float32, [n_batch, h_dim],
                         name="init_c1")
init_h2 = tf.placeholder(tf.float32, [n_batch, h_dim],
                         name="init_h2")
init_c2 = tf.placeholder(tf.float32, [n_batch, h_dim],
                         name="init_c2")

y_tm1 = y_pen_tm1
y_t = y_pen_t

forward_init = "truncated_normal"
norm_clip = True
if norm_clip:
    grad_clip = 3.
else:
    grad_clip = 100.

def step(inp_t,
         att_h_tm1, att_c_tm1,
         h1_tm1, c1_tm1,
         h2_tm1, c2_tm1,
         att_w_tm1, att_k_tm1):
    r = GaussianAttentionCell([inp_t], [3],
                              (att_h_tm1, att_c_tm1),
                              att_k_tm1,
                              X_char,
                              n_letters,
                              h_dim,
                              att_w_tm1,
                              attention_scale=1. / 25.,
                              name="att",
                              random_state=random_state)

    att_w_t = r[0]
    att_k_t = r[1]
    att_phi_t = r[2]
    state = r[-1]

    att_h_t = state[0]
    att_c_t = state[1]

    h1_out_t, state = LSTMCell([inp_t, att_w_t, att_h_t],
                               [3, n_letters, h_dim],
                               h1_tm1, c1_tm1, h_dim,
                               random_state=random_state, name="h1")
    h1_t = state[0]
    c1_t = state[1]

    h2_out_t, state = LSTMCell([inp_t, att_w_t, h1_out_t],
                               [3, n_letters, h_dim],
                               h2_tm1, c2_tm1, h_dim,
                               random_state=random_state, name="h2")
    h2_t = state[0]
    c2_t = state[1]
    return h2_out_t, att_h_t, att_c_t, h1_t, c1_t, h2_t, c2_t, att_w_t, att_k_t, att_phi_t

outputs_info = [None,
                init_att_h,
                init_att_c,
                init_h1,
                init_c1,
                init_h2,
                init_c2,
                init_att_w, init_att_k, None]

o = scan(step, [y_tm1], outputs_info)

#return h2_out_t, att_h_t, att_c_t, h1_t, c1_t, h2_t, c2_t, att_w_t, att_k_t, att_phi_t
sout = o[0]
att_h = o[1]
att_c = o[2]
h1 = o[3]
c1 = o[4]
h2 = o[5]
c2 = o[6]
att_w = o[7]
att_k = o[8]
att_phi = o[9]

att_h = tf.identity(att_h, name="att_h")
att_c = tf.identity(att_c, name="att_c")
att_k = tf.identity(att_k, name="att_k")
att_w = tf.identity(att_w, name="att_w")
att_phi = tf.identity(att_phi, name="att_phi")
h1 = tf.identity(h1, name="h1")
c1 = tf.identity(c1, name="c1")
h2 = tf.identity(h2, name="h2")
c2 = tf.identity(c2, name="c2")

p = LogitBernoulliAndCorrelatedLogitGMM([sout],
                                        [h_dim],
                                        name="b_gmm",
                                        n_components=n_mdn,
                                        random_state=random_state,
                                        init=forward_init)
logit_bernoullis = p[0]
logit_coeffs = p[1]
mus = p[2]
logit_sigmas = p[3]
corrs = p[4]

cost = LogitBernoulliAndCorrelatedLogitGMMCost(
    logit_bernoullis, logit_coeffs, mus, logit_sigmas, corrs, y_t, name="cost")
#masked_cost = y_mask_t * cost
#loss = tf.reduce_mean(masked_cost)
#loss = tf.reduce_sum(masked_cost / (tf.reduce_sum(y_mask_t) + 1E-6))
loss = tf.reduce_mean(cost)
loss = tf.identity(loss, name="loss")

params_dict = get_params_dict()
params = params_dict.values()
grads = tf.gradients(loss, params)

if norm_clip:
    grads, _ = tf.clip_by_global_norm(grads, grad_clip)
else:
    grads = [tf.clip_by_value(grad, -grad_clip, grad_clip) for grad in grads]

#learning_rate = 0.0002
steps = tf.Variable(0.)
learning_rate = tf.train.exponential_decay(0.001, steps, staircase=True, decay_steps=10000,
        decay_rate=0.5)

opt = tf.train.AdamOptimizer(learning_rate=learning_rate, use_locking=True)
updates = opt.apply_gradients(zip(grads, params), global_step=steps)
#updates = opt.apply_gradients(zip(grads, params))

summary = tf.summary.merge([
    tf.summary.scalar('loss', loss),
])

#def loop(itr, sess, extras):
def loop(bgitr, sess, inits, extras):
    trace_mb, text_mb, reset, needed = bg.next_batch2()
    trace_mb = trace_mb.transpose(1, 0, 2)
    text_mb = text_mb.transpose(1, 0, 2)

    # make better masks...
    trace_mask = np.ones_like(trace_mb[:, :, 0])
    text_mask = np.ones_like(text_mb[:, :, 0])
    trace_last_step = trace_mb.shape[0] * trace_mb[0, :, 0]
    for mbi in range(trace_mb.shape[1]):
        for step in range(trace_mb.shape[0]):
            if trace_mb[step:, mbi].min() == 0. and trace_mb[step:, mbi].max() == 0.:
                trace_last_step[mbi] = step
                trace_mask[step:, mbi] = 0.
                break

    text_last_step = text_mb.shape[0] * text_mb[0, :, 0]
    for mbi in range(text_mb.shape[1]):
        for step in range(text_mb.shape[0]):
            if text_mb[step:, mbi].min() == 0. and text_mb[step:, mbi].max() == 0.:
                text_last_step[mbi] = step
                text_mask[step:, mbi] = 0.
                break

    # they don't use good masking? fine
    text_mask = 0. * text_mask + 1.
    trace_mask = 0. * trace_mask + 1.

    #trace_mb, trace_mask, text_mb, text_mask = next(itr)
    init_h1_np = inits[0]
    init_c1_np = inits[1]
    init_h2_np = inits[2]
    init_c2_np = inits[3]
    init_att_h_np = inits[4]
    init_att_c_np = inits[5]
    init_att_k_np = inits[6]
    init_att_w_np = inits[7]

    if not needed:
        do_reset = np.where(reset == 1.)[0]
        init_h1_np[do_reset] = 0. * init_h1_np[do_reset]
        init_c1_np[do_reset] = 0. * init_c1_np[do_reset]
        init_h2_np[do_reset] = 0. * init_h2_np[do_reset]
        init_c2_np[do_reset] = 0. * init_c2_np[do_reset]
        init_att_h_np[do_reset] = 0. * init_att_h_np[do_reset]
        init_att_c_np[do_reset] = 0. * init_att_c_np[do_reset]
        init_att_k_np[do_reset] = 0. * init_att_k_np[do_reset]
        init_att_w_np[do_reset] = 0. * init_att_w_np[do_reset]

    #tbptt_batches = make_tbptt_batches([trace_mb, trace_mask], cut_len)
    feed = {X_char: text_mb,
            y_pen_tm1: trace_mb[:-1],
            y_pen_t: trace_mb[1:],
            init_h1: init_h1_np,
            init_c1: init_c1_np,
            init_h2: init_h2_np,
            init_c2: init_c2_np,
            init_att_h: init_att_h_np,
            init_att_c: init_att_c_np,
            init_att_k: init_att_k_np,
            init_att_w: init_att_w_np}

    if extras["train"]:
        outs = [loss, summary, updates,
                h1, c1,
                h2, c2,
                att_h, att_c,
                att_k, att_w]
        p = sess.run(outs, feed)
        train_loss = p[0]
        train_summary = p[1]
        _ = p[2]
        h1_np = p[3]
        c1_np = p[4]
        h2_np = p[5]
        c2_np = p[6]
        att_h_np = p[7]
        att_c_np = p[8]
        att_k_np = p[9]
        att_w_np = p[10]
        lasts = [h1_np[-1],
                 c1_np[-1],
                 h2_np[-1],
                 c2_np[-1],
                 att_h_np[-1],
                 att_c_np[-1],
                 att_k_np[-1],
                 att_w_np[-1]]
    else:
        outs = [loss]
        p = sess.run(outs, feed)
        train_loss = p[0]
        train_summary = None
        lasts = []
    return [train_loss, train_summary, lasts]

sess1 = tf.Session()
with sess1.as_default() as sess:
    tf.global_variables_initializer().run()
    print_network(get_params_dict())
    model_saver = tf.train.Saver(max_to_keep=2)
    experiment_path = next_experiment_path()
    print("Using experiment path {}".format(experiment_path))
    shutil.copy2(os.getcwd() + "/" + __file__, experiment_path)

    global_step = 0
    summary_writer = tf.summary.FileWriter(experiment_path, flush_secs=10)
    summary_writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START),
                                   global_step=global_step)

    init_h1_np = np.zeros((n_batch, h_dim)).astype("float32")
    init_c1_np = np.zeros((n_batch, h_dim)).astype("float32")
    init_h2_np = np.zeros((n_batch, h_dim)).astype("float32")
    init_c2_np = np.zeros((n_batch, h_dim)).astype("float32")
    init_att_h_np = np.zeros((n_batch, h_dim)).astype("float32")
    init_att_c_np = np.zeros((n_batch, h_dim)).astype("float32")
    init_att_k_np = np.zeros((n_batch, n_attention)).astype("float32")
    init_att_w_np = np.zeros((n_batch, n_letters)).astype("float32")

    inits = [init_h1_np,
             init_c1_np,
             init_h2_np,
             init_c2_np,
             init_att_h_np,
             init_att_c_np,
             init_att_k_np, init_att_w_np]

    bg = BatchGenerator(n_batch, cut_len, random_seed=iteration_seed)
    total_batch_max = n_epochs * 1000

    e = 0
    next_inits = inits
    print(" ")
    print("Training started")
    for bi in range(total_batch_max):
        ret = loop(bg, sess, next_inits, {"train": True})
        train_loss = ret[0]
        train_summary = ret[1]
        next_inits = ret[-1]
        summary_writer.add_summary(train_summary, global_step=bi)
        if bi % 1000 == 0:
            print(" ")
            print("Epoch {}".format(e))
            print(" ")
            model_saver.save(sess, os.path.join(experiment_path, 'models', 'model'), global_step=e)
            e += 1
        print('\r[{:5d}/{:5d}] loss = {}'.format(bi % 1000, 1000, train_loss), end="")

    # save the very last one...
    model_saver.save(sess, os.path.join(experiment_path, 'models', 'model'), global_step=e)
    e += 1

    init_h1_np = np.zeros((n_batch, h_dim)).astype("float32")
    init_c1_np = np.zeros((n_batch, h_dim)).astype("float32")
    init_h2_np = np.zeros((n_batch, h_dim)).astype("float32")
    init_c2_np = np.zeros((n_batch, h_dim)).astype("float32")
    init_att_h_np = np.zeros((n_batch, h_dim)).astype("float32")
    init_att_c_np = np.zeros((n_batch, h_dim)).astype("float32")
    init_att_k_np = np.zeros((n_batch, n_attention)).astype("float32")
    init_att_w_np = np.zeros((n_batch, n_letters)).astype("float32")

    inits = [init_h1_np,
             init_c1_np,
             init_h2_np,
             init_c2_np,
             init_att_h_np,
             init_att_c_np,
             init_att_k_np, init_att_w_np]

    print("")
    bg = BatchGenerator(n_batch, cut_len, random_seed=iteration_seed)
    for i in range(10):
        ret = loop(bg, sess, inits, {"train": False})
        print(ret[0])

    print("")
    bg = BatchGenerator(n_batch, cut_len, random_seed=iteration_seed)
    for i in range(10):
        ret = loop(bg, sess, inits, {"train": False})
        print(ret[0])
