import matplotlib
matplotlib.use("Agg")

import argparse
import tensorflow as tf
import numpy as np
from tfbldr.datasets import make_sinewaves
from collections import namedtuple, defaultdict
import sys
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('direct_model', nargs=1, default=None)
parser.add_argument('--model', dest='model_path', type=str, default=None)
parser.add_argument('--seed', dest='seed', type=int, default=1999)
args = parser.parse_args()
if args.model_path == None:
    if args.direct_model == None:
        raise ValueError("Must pass first positional argument as model, or --model argument, e.g. summary/experiment-0/models/model-7")
    else:
        model_path = args.direct_model[0]
else:
    model_path = args.model_path

sines = make_sinewaves(50, 40, square=True)
train_sines = sines[:, ::2]
train_sines = [train_sines[:, i] for i in range(train_sines.shape[1])]
valid_sines = sines[:, 1::2]
valid_sines = [valid_sines[:, i] for i in range(valid_sines.shape[1])]

random_state = np.random.RandomState(args.seed)

config = tf.ConfigProto(
    device_count={'GPU': 0}
)

n_hid = 100
batch_size = 10

with tf.Session(config=config) as sess:
    saver = tf.train.import_meta_graph(model_path + '.meta')
    saver.restore(sess, model_path)
    fields = ["inputs",
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
              "loss",
              "rec_loss",
              "train_step"]
    vs = namedtuple('Params', fields)(
        *[tf.get_collection(name)[0] for name in fields]
    )
    x = np.array(valid_sines)
    x = x.transpose(1, 0, 2)
    prev_x = 0. * x[:1, :batch_size] + 1.
    res = []
    q_res = []
    h_res = []
    i_res = []
    init_h = np.zeros((batch_size, n_hid))
    init_c = np.zeros((batch_size, n_hid))
    init_q_h = np.zeros((batch_size, n_hid)).astype("float32")
    init_q_c = np.zeros((batch_size, n_hid)).astype("float32")
    for i in range(50):
        feed = {vs.inputs_tm1: prev_x[:1],
                vs.init_hidden: init_h,
                vs.init_cell: init_c,
                vs.init_q_hidden: init_q_h,
                vs.init_q_cell: init_q_c}
        outs = [vs.pred, vs.hiddens, vs.cells, vs.q_hiddens, vs.q_cells, vs.i_hiddens]
        r = sess.run(outs, feed_dict=feed)
        prev_x = r[0]
        hids = r[1]
        cs = r[2]
        q_hids = r[3]
        q_cs = r[4]
        i_hids = r[5]
        init_h = hids[0]
        init_c = cs[0]
        init_q_h = q_hids[0]
        init_q_c = q_cs[0]
        res.append(prev_x)
        q_res.append(q_hids)
        h_res.append(hids)
        i_res.append(i_hids)
    o = np.concatenate(res, axis=0)[:, :, 0]
    ind = np.concatenate(i_res, axis=0)

    f, axarr = plt.subplots(11, 1)
    for i in range(10):
        if i % 2 == 0:
            axarr[i].plot(o[:, i // 2])
        else:
            axarr[i].plot(ind[:, i // 2])

    axarr[-1].plot(valid_sines[0][:, 0], color="r")

    plt.savefig("results")
    plt.close()
