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

cut = 512
step = 512

train_d = np.load("stage1_data/train_data.npz")
valid_d = np.load("stage1_data/valid_data.npz")

train_arr = train_d["z_i_x"]
valid_arr = valid_d["z_i_x"]
train_data = [train_arr[i].transpose(1, 0) for i in range(len(train_arr))]
valid_data = [valid_arr[i].transpose(1, 0) for i in range(len(valid_arr))]

lu = {tuple(k.ravel()): v for v, k in enumerate(train_d["all_keys"])}

train_i_data = [lu[tuple(train_data[i].ravel())] for i in range(len(train_data))]
valid_i_data = [lu[tuple(valid_data[i].ravel())] for i in range(len(valid_data))]

# make overlapping chunks of 10, overlapped by 5
# not ideal, more ideally we could make this variable length, per word
def list_overlap(l, size=10, step=5):
    l = l[:len(l) - len(l) % step]
    finals = []
    ss = np.arange(0, len(l) - size + step, step)
    for sis in ss:
        finals.append(l[sis:sis+size])
    return finals

tid = list_overlap(train_i_data)
tid = np.array(tid).astype("float32")
tid = tid[..., None]
vid = list_overlap(valid_i_data)
vid = np.array(vid).astype("float32")
vid = vid[..., None]

batch_size = 10
n_z_clusters = int(tid.max())
n_hid = 512
inner_seq_len = tid.shape[1]
rounds = 7

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
    x = tid.transpose(1, 0, 2)[0]
    x = 10 * np.arange(batch_size)[None, :, None] + 0. * x[None, :batch_size]
    init_h = np.zeros((batch_size, n_hid)).astype("float32")
    init_c = np.zeros((batch_size, n_hid)).astype("float32")
    init_q_h = np.zeros((batch_size, n_hid)).astype("float32")
    init_q_c = np.zeros((batch_size, n_hid)).astype("float32")
    res = []
    i_res = []
    for n_round in range(rounds):
        this_res = [x]
        this_i_res = [-1 * init_q_h[:, 0][None].astype("float32")]
        # reset these, trained only up to inner_seq_len dynamics get terrible
        init_h = np.zeros((batch_size, n_hid)).astype("float32")
        init_c = np.zeros((batch_size, n_hid)).astype("float32")
        init_q_h = np.zeros((batch_size, n_hid)).astype("float32")
        init_q_c = np.zeros((batch_size, n_hid)).astype("float32")
        for i in range(inner_seq_len - 1):
            feed = {vs.inputs_tm1: x,
                    vs.init_hidden: init_h,
                    vs.init_cell: init_c,
                    vs.init_q_hidden: init_q_h,
                    vs.init_q_cell: init_q_c}
            outs = [vs.pred_sm, vs.hiddens, vs.cells, vs.q_hiddens, vs.q_cells, vs.i_hiddens]
            r = sess.run(outs, feed_dict=feed)
            p = r[0]
            # sample?
            #x = np.array([sample_random_state.choice(list(range(n_z_clusters)), p=p[0, i]) for i in range(batch_size)]).astype("float32")[None, :, None]
            x = p.argmax(axis=-1)[:, :, None].astype("float32")
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
        x = x + 1. + 90.
    final_quantized_indices = np.array(res)
    final_hidden_indices = np.array(i_res)

# 7, 10, 1, 10, 1 -> 7, 10, 10 in the right order
quantized = final_quantized_indices[:, :, 0, :, 0].astype("int32").transpose(0, 2, 1)
indices = final_hidden_indices[:, :, 0].transpose(0, 2, 1)
quantized = quantized.reshape(-1, quantized.shape[-1])
indices = indices.reshape(-1, indices.shape[-1])
r_lu = {v: k for k, v in lu.items()}
# need to look these back up into something...
q_shp = quantized.shape
codes = [r_lu[q] for q in quantized.ravel().astype("int32")]
codes = np.array(codes)[:, None]
#codes = np.array(codes).reshape((q_shp[0], q_shp[1], -1))
# make it look right for vq vae
final_quantized_indices = codes.astype("int64")

final_hidden_indices = indices.flatten()[:, None]

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
    i_buf[ni * step:(ni * step) + cut] = i_rec[ni]

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
