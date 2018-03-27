import matplotlib
matplotlib.use("Agg")
import os

import argparse
import tensorflow as tf
import numpy as np
from tfbldr.datasets import fetch_fruitspeech
from tfbldr.datasets.audio import soundsc
from tfbldr.datasets.audio import overlap
from tfbldr.plot import specgram
from tfbldr.plot import specplot
import copy

from collections import namedtuple, defaultdict
import sys
import matplotlib.pyplot as plt
from scipy.io import wavfile


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

random_state = np.random.RandomState(args.seed)

config = tf.ConfigProto(
    device_count={'GPU': 0}
)

fruit = fetch_fruitspeech()
minmin = np.inf
maxmax = -np.inf

for s in fruit["data"]:
    minmin = min(minmin, s.min())
    maxmax = max(maxmax, s.max())

train_data = []
valid_data = []
type_counts = defaultdict(lambda: 0)
final_audio = []
for n, s in enumerate(fruit["data"]):
    type_counts[fruit["target"][n]] += 1
    n_s = (s - minmin) / float(maxmax - minmin)
    n_s = 2 * n_s - 1
    if type_counts[fruit["target"][n]] == 15:
        valid_data.append(n_s)
    else:
        train_data.append(n_s)

# no overlap for now
cut = 256
step = 256
train_data = np.concatenate(train_data, axis=0)
valid_data = np.concatenate(valid_data, axis=0)
train_audio = overlap(train_data, cut, step)
valid_audio = overlap(valid_data, cut, step)

with tf.Session(config=config) as sess:
    saver = tf.train.import_meta_graph(model_path + '.meta')
    saver.restore(sess, model_path)
    fields = ['images',
              'bn_flag',
              'z_e_x',
              'z_q_x',
              'z_i_x',
              'x_tilde']
    vs = namedtuple('Params', fields)(
        *[tf.get_collection(name)[0] for name in fields]
    )
    if not os.path.exists("stage1_data"):
        os.mkdir("stage1_data")
    for n, aa in enumerate([train_audio, valid_audio]):
        x = aa[:, None, :, None]
        feed = {vs.images: x,
                vs.bn_flag: 1.}
        outs = [vs.z_e_x, vs.z_q_x, vs.z_i_x, vs.x_tilde]
        r = sess.run(outs, feed_dict=feed)
        z_e_x = r[0]
        z_q_x = r[1]
        z_i_x = r[2]
        x_rec = r[-1]
        d = {"in_x": x,
             "z_e_x": z_e_x,
             "z_q_x": z_q_x,
             "z_i_x": z_i_x,
             "x_rec": x_rec}
        name = "stage1_data/{}"
        if n == 0:
            name = name.format("train_data")
        else:
            name = name.format("valid_data")
        np.savez(name, **d)

train_d = np.load("stage1_data/train_data.npz")
train_d = {k:v for k, v in train_d.items()}
valid_d = np.load("stage1_data/valid_data.npz")
valid_d = {k:v for k, v in valid_d.items()}

train_arr = train_d["z_i_x"]
valid_arr = valid_d["z_i_x"]
train_keys = [tuple(train_arr[i, 0].ravel()) for i in range(len(train_arr[:, 0]))]
valid_keys = [tuple(valid_arr[i, 0].ravel()) for i in range(len(valid_arr[:, 0]))]
all_keys = train_keys + valid_keys
train_d["train_keys"] = np.array(train_keys)
train_d["valid_keys"] = np.array(valid_keys)
train_d["all_keys"] = np.array(all_keys)

valid_d["train_keys"] = np.array(train_keys)
valid_d["valid_keys"] = np.array(valid_keys)
valid_d["all_keys"] = np.array(all_keys)

base_name = "stage1_data/{}"
name = base_name.format("train_data")
np.savez(name, **train_d)
name = base_name.format("valid_data")
np.savez(name, **valid_d)
