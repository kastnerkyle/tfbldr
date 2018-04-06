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

sines = make_sinewaves(50, 40)
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
              "hiddens",
              "pred",
              "loss",
              "rec_loss",
              "train_step"]
    vs = namedtuple('Params', fields)(
        *[tf.get_collection(name)[0] for name in fields]
    )
    x = np.array(valid_sines)
    x = x.transpose(1, 0, 2)
    prev_x = x[:1, :batch_size]
    res = []
    init_h = np.zeros((batch_size, n_hid))
    for i in range(50):
        feed = {vs.inputs_tm1: prev_x[:1],
                vs.init_hidden: init_h}
        outs = [vs.pred, vs.hiddens]
        r = sess.run(outs, feed_dict=feed)
        prev_x = r[0]
        hids = r[1]
        res.append(prev_x)
        init_h = hids[0]
    o = np.concatenate(res, axis=0)[:, :, 0]

    f, axarr = plt.subplots(5, 1)
    for i in range(5):
        axarr[i].plot(o[:, i])

    plt.savefig("results")
    plt.close()
