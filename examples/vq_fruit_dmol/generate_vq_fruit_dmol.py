import matplotlib
matplotlib.use("Agg")

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

cut = 256
# change to no overlap for now
step = 16
n_components = 5
eval_batch_size = 500
train_data = np.concatenate(train_data, axis=0)
valid_data = np.concatenate(valid_data, axis=0)
train_audio = overlap(train_data, cut, step)
valid_audio = overlap(valid_data, cut, step)

sample_random_state = np.random.RandomState(1122)

with tf.Session(config=config) as sess:
    saver = tf.train.import_meta_graph(model_path + '.meta')
    saver.restore(sess, model_path)
    fields = ['images',
              'bn_flag',
              'z_e_x',
              'z_q_x',
              'z_i_x',
              'x_tilde_mix',
              'x_tilde_means',
              'x_tilde_lin_scales']
    for field in fields:
        print(field)
        tf.get_collection(field)[0]
    vs = namedtuple('Params', fields)(
        *[tf.get_collection(name)[0] for name in fields]
    )
    all_x = valid_audio[:, None, :, None]
    all_x_rec = []
    print("Finished restoring parameters, running audio of size {}".format(all_x.shape))
    start_inds = np.arange(0, len(all_x), eval_batch_size)
    for n, i in enumerate(start_inds):
        x = all_x[i:i + eval_batch_size]
        print("Running eval batch {} of {}, size {}".format(n + 1, len(start_inds), x.shape))
        feed = {vs.images: x,
                vs.bn_flag: 1.}
        outs = [vs.z_e_x, vs.z_q_x, vs.z_i_x, vs.x_tilde_mix, vs.x_tilde_means, vs.x_tilde_lin_scales]
        r = sess.run(outs, feed_dict=feed)
        x_rec_mix = r[-3]
        x_rec_means = r[-2]
        x_rec_lin_scales = r[-1]

        shp = x_rec_means.shape
        x_rec_lin_scales = np.maximum(x_rec_lin_scales, -7)

        # gumbel sample
        # http://amid.fish/humble-gumbel
        x_rec_samp_mix = np.argmax(x_rec_mix - np.log(-np.log(sample_random_state.uniform(low=1E-5, high=1-1E-5, size=x_rec_mix.shape))), axis=-1)
        x_rec_samp_means = x_rec_means.reshape((-1, shp[-1]))
        x_rec_samp_means = x_rec_samp_means[np.arange(len(x_rec_samp_means)), x_rec_samp_mix.flatten()].reshape(shp[:-1])

        x_rec_samp_lin_scales = x_rec_lin_scales.reshape((-1, shp[-1]))
        x_rec_samp_lin_scales = x_rec_samp_lin_scales[np.arange(len(x_rec_samp_lin_scales)), x_rec_samp_mix.flatten()].reshape(shp[:-1])

        u = sample_random_state.uniform(low=1E-5, high=1-1E-5, size=x_rec_samp_means.shape)
        x_rec = x_rec_samp_means + x_rec_samp_lin_scales * (np.log(u) - np.log(1 - u))
        from IPython import embed; embed(); raise ValueError()

        all_x_rec.append(x_rec)

    x = all_x
    x_rec = np.concatenate(all_x_rec, axis=0)

    rec_buf = np.zeros((len(x_rec) * step + 2 * cut))
    for ni in range(len(x_rec)):
        t = x_rec[ni, 0]
        t = t[:, 0]
        rec_buf[ni * step:(ni * step) + cut] += t

    orig_buf = np.zeros((len(x) * step + 2 * cut))
    for ni in range(len(x)):
        t = x[ni, 0]
        t = t[:, 0]
        orig_buf[ni * step:(ni * step) + cut] += t

    x_o = orig_buf
    x_r = rec_buf

    f, axarr = plt.subplots(2, 1)
    axarr[0].plot(x_r)
    axarr[0].set_title("Reconstruction")
    axarr[1].plot(x_o)
    axarr[1].set_title("Original")

    plt.savefig("vq_vae_generation_results")
    plt.close()

    f, axarr = plt.subplots(2, 1)
    specplot(specgram(x_r), axarr[0])
    axarr[0].set_title("Reconstruction")
    specplot(specgram(x_o), axarr[1])
    axarr[1].set_title("Original")

    plt.savefig("vq_vae_generation_results_spec")
    plt.close()

    wavfile.write("original_wav.wav", 8000, soundsc(x_o))
    wavfile.write("reconstructed_wav.wav", 8000, soundsc(x_r))
    from IPython import embed; embed(); raise ValueError()
