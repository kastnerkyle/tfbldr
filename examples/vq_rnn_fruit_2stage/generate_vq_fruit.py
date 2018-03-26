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

from collections import namedtuple, defaultdict
import sys
import matplotlib.pyplot as plt
from scipy.io import wavfile


parser = argparse.ArgumentParser()
parser.add_argument('stage1_direct_model', nargs=1, default=None)
parser.add_argument('stage2_direct_model', nargs=1, default=None)
parser.add_argument('--seed', dest='seed', type=int, default=1999)
args = parser.parse_args()
stage1_direct_model = args.stage1_direct_model[0]
stage2_direct_model = args.stage2_direct_model[0]

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

cut = 512
step = 512
train_data = np.concatenate(train_data, axis=0)
valid_data = np.concatenate(valid_data, axis=0)
train_audio = overlap(train_data, cut, step)
valid_audio = overlap(valid_data, cut, step)

train_d = np.load("stage1_data/train_data.npz")
batch_size = 10
n_z_clusters = 512
n_hid = 512
inner_seq_len = 32
rounds = 5

sample_random_state = np.random.RandomState(1165)

with tf.Session(config=config) as sess:
    saver = tf.train.import_meta_graph(stage2_direct_model + '.meta')
    saver.restore(sess, stage2_direct_model)
    fields = ['inputs_tm1',
              'init_hidden',
              'init_cell',
              'init_q_hidden',
              'init_q_cell',
              'hiddens',
              'cells',
              'q_hiddens',
              'q_cells',
              'i_hiddens',
              'pred',
              'pred_sm']
    vs = namedtuple('Params', fields)(
        *[tf.get_collection(name)[0] for name in fields]
    )
    # use compressed from train as start seeds...
    x = train_d["z_i_x"][:, 0, 0]
    x = x[None, :]
    x = 0. * x[:, :batch_size, None]
    init_h = np.zeros((batch_size, n_hid)).astype("float32")
    init_c = np.zeros((batch_size, n_hid)).astype("float32")
    init_q_h = np.zeros((batch_size, n_hid)).astype("float32")
    init_q_c = np.zeros((batch_size, n_hid)).astype("float32")
    res = []
    i_res = []
    for n_round in range(rounds):
        this_res = [x]
        this_i_res = [-1 * init_q_h[:, 0][None].astype("float32")]
        for i in range(inner_seq_len - 1):
            # cheater
            feed = {vs.inputs_tm1: x,
                    vs.init_hidden: init_h,
                    vs.init_cell: init_c,
                    vs.init_q_hidden: init_q_h,
                    vs.init_q_cell: init_q_c}
            outs = [vs.pred_sm, vs.hiddens, vs.cells, vs.q_hiddens, vs.q_cells, vs.i_hiddens]
            r = sess.run(outs, feed_dict=feed)
            p = r[0]
            # sample?
            x = np.array([sample_random_state.choice(list(range(n_z_clusters)), p=p[0, i]) for i in range(batch_size)]).astype("float32")[None, :, None]
            #x = p.argmax(axis=-1)[:, :, None].astype("float32")
            hids = r[1]
            cs = r[2]
            q_hids = r[3]
            q_cs = r[4]
            i_hids = r[5]
            init_h = hids[0]
            init_c = cs[0]
            init_q_h = q_hids[0]
            init_q_c = q_cs[0]
            this_res.append(x)
            this_i_res.append(i_hids)
        res.append(this_res)
        i_res.append(this_i_res)
    final_quantized_indices = np.array(res)
    final_hidden_indices = np.array(i_res)

# make it look right for vq vae
final_quantized_indices = final_quantized_indices.transpose(0, 2, 3, 4, 1)
final_quantized_indices = final_quantized_indices.reshape((-1, inner_seq_len))
final_quantized_indices = final_quantized_indices[:, None]
final_quantized_indices = final_quantized_indices.astype("int64")

final_hidden_indices = final_hidden_indices.transpose(0, 2, 3, 1)
final_hidden_indices = final_hidden_indices.reshape((-1, inner_seq_len))
final_hidden_indices = final_hidden_indices[:, None]
final_hidden_indices = final_hidden_indices.astype("int64")

tf.reset_default_graph()

with tf.Session(config=config) as sess2:
    saver = tf.train.import_meta_graph(stage1_direct_model + '.meta')
    saver.restore(sess2, stage1_direct_model)
    fields = ['images',
              'bn_flag',
              'z_e_x',
              'z_q_x',
              'z_i_x',
              'x_tilde']
    vs = namedtuple('Params', fields)(
        *[tf.get_collection(name)[0] for name in fields]
    )

    feed = {vs.images: np.zeros((len(final_quantized_indices), 1, 512, 1)).astype("float32"),
            vs.z_i_x: final_quantized_indices,
            vs.bn_flag: 1.}
    outs = [vs.x_tilde]
    r = sess2.run(outs, feed_dict=feed)
    x_rec = r[0]

i_rec = final_hidden_indices
rec_buf = np.zeros((len(x_rec) * step + 2 * cut))
i_buf = np.zeros((len(x_rec) * step + 2 * cut))
for ni in range(len(x_rec)):
    t = x_rec[ni, 0]
    t = t[:, 0]
    rec_buf[ni * step:(ni * step) + cut] += t
    for ii in range(inner_seq_len):
        i_buf[ni * step + ii * inner_seq_len:ni * step + (ii + 1) * inner_seq_len] = i_rec[ni, 0, ii]

x_r = rec_buf
i_r = i_buf

f, axarr = plt.subplots(2, 1)
axarr[0].plot(x_r)
axarr[0].set_title("Generation")
axarr[1].plot(i_r, color="r")

plt.savefig("vq_vae_generation_results")
plt.close()

f, axarr = plt.subplots(1, 1)
specplot(specgram(x_r), axarr)
axarr.set_title("Generation")
#axarr[1].plot(i_r, color="r")

plt.savefig("vq_vae_generation_results_spec")
plt.close()

wavfile.write("generated_wav.wav", 8000, soundsc(x_r))
