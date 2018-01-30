from __future__ import print_function
from extras import fetch_iamondb, rsync_fetch, list_iterator
from tfdllib import Linear, scan, get_params_dict, print_network
import tensorflow as tf
import numpy as np

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
h_dim = 256
grad_clip = 100.
n_batch = trace_mb.shape[1]

X_char = tf.placeholder(tf.float32, shape=[None, n_batch, n_letters])
y_pen = tf.placeholder(tf.float32, shape=[None, n_batch, 3])
init_h1 = tf.placeholder(tf.float32, [n_batch, h_dim])
bias = tf.placeholder_with_default(tf.zeros(shape=[]), shape=[])

y_tm1 = y_pen[:-1, :, :]
y_t = y_pen[1:, :, :]

proj_inp = Linear([y_tm1], [3], h_dim, random_state=random_state,
                   weight_norm=False)


def step(inp_t, h1_tm1):
    h1_t_proj = Linear([inp_t], [h_dim], h_dim, random_state=random_state,
                       weight_norm=False)
    ret = h1_t_proj + h1_tm1
    return ret

h1_o = scan(step, [proj_inp], [init_h1])

cost = tf.reduce_mean(h1_o)
params_dict = get_params_dict()
params = params_dict.values()
grads = tf.gradients(cost, params)

learning_rate = .0002
grads = [tf.clip_by_value(grad, -grad_clip, grad_clip) for grad in grads]
opt = tf.train.AdamOptimizer(learning_rate, use_locking=True)
updates = opt.apply_gradients(zip(grads, params))

def loop(itr, sess, extras):
    trace_mb, trace_mask, text_mb, text_mask = next(itr)
    init_h1_np = np.zeros((n_batch, h_dim)).astype("float32")
    feed = {X_char: text_mb,
            y_pen: trace_mb,
            init_h1: init_h1_np}
    if extras["train"]:
        outs = [cost, updates]
        train_loss, _ = sess.run(outs, feed)
    else:
        outs = [cost,]
        train_loss = sess.run(outs, feed)[0]
    return


sess = tf.Session()

with sess.as_default():
    tf.global_variables_initializer().run()
    print_network(get_params_dict())
    av = tf.global_variables()
    try:
        while True:
            loop(train_itr, sess, {"train": True})
            print(".", end="")
    except StopIteration:
        print(" ")
        print("Training done")
        train_itr.reset()

    try:
        while True:
            loop(train_itr, sess, {"train": False})
            print(".", end="")
    except StopIteration:
        print(" ")
        print("Validation done")
        train_itr.reset()
