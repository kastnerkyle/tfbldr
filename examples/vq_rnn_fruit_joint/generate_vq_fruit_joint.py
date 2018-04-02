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
from scipy.io import wavfile

from tfbldr.plot import get_viridis
viridis_cm = get_viridis()
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('direct_model', nargs=1, default=None)
parser.add_argument('--seed', dest='seed', type=int, default=1999)
args = parser.parse_args()
direct_model = args.direct_model[0]

random_state = np.random.RandomState(args.seed)

config = tf.ConfigProto(
    device_count={'GPU': 0}
)

fruit = fetch_fruitspeech()
minmin = np.inf
maxmax = -np.inf

for s in fruit["data"]:
    si = s - s.mean()
    minmin = min(minmin, si.min())
    maxmax = max(maxmax, si.max())

train_data = []
valid_data = []
type_counts = defaultdict(lambda: 0)
final_audio = []

for n, s in enumerate(fruit["data"]):
    type_counts[fruit["target"][n]] += 1
    s = s - s.mean()
    n_s = (s - minmin) / float(maxmax - minmin)
    n_s = 2 * n_s - 1
    #n_s = mu_law_transform(n_s, 256)
    if type_counts[fruit["target"][n]] == 15:
        valid_data.append(n_s)
    else:
        train_data.append(n_s)


def _cuts(list_of_audio, cut, step):
    # make many overlapping cuts
    # 8k, this means offset is ~4ms @ step of 32
    real_final = []
    real_idx = []
    for n, s in enumerate(list_of_audio):
        # cut off the end
        s = s[:len(s) - len(s) % step]
        starts = np.arange(0, len(s) - cut + step, step)
        for st in starts:
            real_final.append(s[st:st + cut][None, :, None])
            real_idx.append(n)
    return real_final, real_idx

cut = 256
step = 1
train_audio, train_audio_idx = _cuts(train_data, cut, step)
valid_audio, valid_audio_idx = _cuts(valid_data, cut, step)

embedding_dim = 512
vqvae_batch_size = 50
rnn_batch_size = 50
n_hid = 512
n_clusters = 64
# reserve 0 for "start code"
n_inputs = embedding_dim + 1
hardcoded_z_len = 16
reps = 10
sample_random_state = np.random.RandomState(1165)

with tf.Session(config=config) as sess:
    saver = tf.train.import_meta_graph(direct_model + '.meta')
    saver.restore(sess, direct_model)
    fields = ["vqvae_inputs",
              "bn_flag",
              "x_tilde",
              "z_e_x",
              "z_q_x",
              "z_i_x",
              "z_emb",
              "vqvae_rec_loss",
              "rnn_inputs",
              "rnn_inputs_tm1",
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
              "pred_sm",
              "rnn_rec_loss"]
    vs = namedtuple('Params', fields)(
        *[tf.get_collection(name)[0] for name in fields]
    )
    init_h = np.zeros((rnn_batch_size, n_hid)).astype("float32")
    init_c = np.zeros((rnn_batch_size, n_hid)).astype("float32")
    init_q_h = np.zeros((rnn_batch_size, n_hid)).astype("float32")
    init_q_c = np.zeros((rnn_batch_size, n_hid)).astype("float32")

    rnn_inputs = np.zeros((1, rnn_batch_size, 1))
    all_out = []
    all_h = []
    all_c = []
    all_q_h = []
    all_q_c = []
    all_i_h = []
    for i in range(reps * hardcoded_z_len):
        print("Sampling step {} of {}".format(i + 1, reps * hardcoded_z_len))
        feed = {vs.rnn_inputs_tm1: rnn_inputs,
                vs.init_hidden: init_h,
                vs.init_cell: init_c,
                vs.init_q_hidden: init_q_h,
                vs.init_q_cell: init_q_c}
        outs = [vs.pred_sm, vs.hiddens, vs.cells, vs.q_hiddens, vs.q_cells, vs.i_hiddens]
        r = sess.run(outs, feed_dict=feed)
        pred_sm = r[0]

        pred_samp_i = np.argmax(pred_sm - np.log(-np.log(sample_random_state.uniform(low=1E-5, high=1-1E-5, size=pred_sm.shape))), axis=-1)
        pred_i = pred_samp_i.astype("float32")

        #pred_i = pred_sm.argmax(axis=-1).astype("float32")
        rnn_inputs = pred_i[..., None]
        hiddens = r[1]
        cells = r[2]
        q_hiddens = r[3]
        q_cells = r[4]
        i_hiddens = r[-1]
        all_out.append(rnn_inputs)
        all_h.append(hiddens)
        all_c.append(cells)
        all_q_h.append(q_hiddens)
        all_q_c.append(q_cells)
        all_i_h.append(i_hiddens[..., None])
        init_h = hiddens[-1]
        init_c = cells[-1]
        init_q_h = q_hiddens[-1]
        init_q_c = q_cells[-1]

    s_out = np.concatenate(all_out, axis=0)
    i_h = np.concatenate(all_i_h, axis=0)
    from IPython import embed; embed(); raise ValueError()

    res = []
    i_res = []
    for n_round in range(rounds):
        this_res = [x]
        this_i_res = [-1 * init_q_h[:, 0][None].astype("float32")]
        # can reset these, trained only up to inner_seq_len by seems ok if trained on overlap frames
        #init_h = np.zeros((batch_size, n_hid)).astype("float32")
        #init_c = np.zeros((batch_size, n_hid)).astype("float32")
        #init_q_h = np.zeros((batch_size, n_hid)).astype("float32")
        #init_q_c = np.zeros((batch_size, n_hid)).astype("float32")
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
            #x = np.array([sample_random_state.choice(list(range(p.shape[-1])), p=p[0, i]) for i in range(batch_size)]).astype("float32")[None, :, None]
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
        x = x + 1. + inner_seq_len * batch_size
    final_quantized_indices = np.array(res)
    final_hidden_indices = np.array(i_res)

# n_rounds, 10, 1, 10, 1 -> n_rounds, 10, 10 in the right order
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

    # due to straighthrough impl tensorflow still wants the input, set to all 0 to be sure it has no impact
    feed = {vs.images: np.zeros((len(final_quantized_indices), 1, 512, 1)).astype("float32"),
            vs.z_i_x: final_quantized_indices,
            vs.bn_flag: 1.}
    outs = [vs.x_tilde]
    r = sess2.run(outs, feed_dict=feed)
    x_rec = r[0]

# put all the shorter traces into one long sequence, no overlap add or anything
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
