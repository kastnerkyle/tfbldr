import argparse
import tensorflow as tf
import numpy as np
from tfbldr.datasets import fetch_mnist
from collections import namedtuple
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy
from tfbldr.datasets import fetch_ljspeech
from tfbldr.datasets import wavfile_caching_mel_tbptt_iterator
from scipy.io import wavfile
from tfbldr.datasets.audio import soundsc
from tfbldr.datasets.audio import stft
from tfbldr.datasets.audio import iterate_invert_spectrogram


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

"""
config = tf.ConfigProto(
    device_count={'GPU': 0}
)
"""
ljspeech = fetch_ljspeech()
wavfiles = ljspeech["wavfiles"]
txtfiles = ljspeech["txtfiles"]

batch_size = 64
seq_len = 256
window_mixtures = 10
units = 1024
emb_dim = 15

train_random_state = np.random.RandomState(3122)
train_itr = wavfile_caching_mel_tbptt_iterator(wavfiles, txtfiles, batch_size, seq_len, stop_index=.95, shuffle=True, random_state=train_random_state)
mels, mel_mask, text, text_mask, reset = train_itr.next_masked_batch()

#with tf.Session(config=config) as sess:
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(model_path + '.meta')
    saver.restore(sess, model_path)
    fields = ["mels",
              "mel_mask",
              "in_mels",
              "in_mel_mask",
              "out_mels",
              "out_mel_mask",
              "text",
              "text_mask",
              "bias",
              "cell_dropout",
              "noise",
              "pred",
              #"mix", "means", "lins",
              "att_w_init",
              "att_k_init",
              "att_h_init",
              "att_c_init",
              "h1_init",
              "c1_init",
              "h2_init",
              "c2_init",
              "att_w",
              "att_k",
              "att_phi",
              "att_h",
              "att_c",
              "h1",
              "c1",
              "h2",
              "c2"]
    vs = namedtuple('Params', fields)(
        *[tf.get_collection(name)[0] for name in fields]
    )

    att_w_init = np.zeros((batch_size, emb_dim))
    att_k_init = np.zeros((batch_size, window_mixtures))
    att_h_init = np.zeros((batch_size, units))
    att_c_init = np.zeros((batch_size, units))
    h1_init = np.zeros((batch_size, units))
    c1_init = np.zeros((batch_size, units))
    h2_init = np.zeros((batch_size, units))
    c2_init = np.zeros((batch_size, units))

    # only predict 1 thing
    endpoint = np.where(text_mask[:, 0] == 1)[0][-1]

    """
    text = text[:endpoint]
    for jj in range(text.shape[1]):
        text[:, jj] = text[:, 0]
    text_mask = 0. * text[:, :, 0] + 1.
    """

    in_mels = 0. * mels[:1]
    in_mel_mask = 0. * mel_mask[:1] + 1.

    preds = []
    att_ws = []
    att_phis = []
    random_state = np.random.RandomState(11)
    noise_scale = 4.
    for ii in range(2048):
        print("pred step {}".format(ii))
        noise_block = np.clip(random_state.randn(*in_mels.shape), -6, 6)
        in_mels = in_mels + noise_scale * noise_block
        #if ii > 0:
        #    in_mels[0] = mels[ii - 1]
        feed = {
                vs.in_mels: in_mels,
                vs.in_mel_mask: in_mel_mask,
                vs.text: text,
                vs.text_mask: text_mask,
                vs.cell_dropout: 1.,
                vs.att_w_init: att_w_init,
                vs.att_k_init: att_k_init,
                vs.att_h_init: att_h_init,
                vs.att_c_init: att_c_init,
                vs.h1_init: h1_init,
                vs.c1_init: c1_init,
                vs.h2_init: h2_init,
                vs.c2_init: c2_init}
        outs = [vs.att_w, vs.att_k,
                vs.att_h, vs.att_c,
                vs.h1, vs.c1, vs.h2, vs.c2,
                vs.att_phi, vs.pred]
        r = sess.run(outs, feed_dict=feed)
        att_w_np = r[0]
        att_k_np = r[1]
        att_h_np = r[2]
        att_c_np = r[3]
        h1_np = r[4]
        c1_np = r[5]
        h2_np = r[6]
        c2_np = r[7]
        att_phi_np = r[8]
        pred_np = r[9]

        att_ws.append(att_w_np[0])
        att_phis.append(att_phi_np[0])
        preds.append(pred_np[0])

        # set next inits and input values
        in_mels[0] = pred_np
        att_w_init = att_w_np[-1]
        att_k_init = att_k_np[-1]
        att_h_init = att_h_np[-1]
        att_c_init = att_c_np[-1]
        h1_init = h1_np[-1]
        c1_init = c1_np[-1]
        h2_init = h2_np[-1]
        c2_init = c2_np[-1]

preds = np.array(preds)
n_plot = 4
f, axarr = plt.subplots(1, n_plot)
for jj in range(n_plot):
    spectrogram = preds[:, jj] * train_itr._std + train_itr._mean
    axarr[jj].imshow(spectrogram)
plt.savefig("sample_spec.png")

sample_rate = 22050
window_size = 512
step = 128
n_mel = 80

def logmel(waveform):
    z = tf.contrib.signal.stft(waveform, window_size, step)
    magnitudes = tf.abs(z)
    filterbank = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mel,
        num_spectrogram_bins=magnitudes.shape[-1].value,
        sample_rate=sample_rate,
        lower_edge_hertz=125.,
        upper_edge_hertz=7800.)
    melspectrogram = tf.tensordot(magnitudes, filterbank, 1)
    return tf.log1p(melspectrogram)

def sonify(spectrogram, samples, transform_op_fn, logscaled=True):
    graph = tf.Graph()
    with graph.as_default():

        noise = tf.Variable(tf.random_normal([samples], stddev=1e-6))

        x = transform_op_fn(noise)
        y = spectrogram

        if logscaled:
            x = tf.expm1(x)
            y = tf.expm1(y)

        x = tf.nn.l2_normalize(x)
        y = tf.nn.l2_normalize(y)
        tf.losses.mean_squared_error(x, y[-tf.shape(x)[0]:])

        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            loss=tf.losses.get_total_loss(),
            var_list=[noise],
            tol=1e-16,
            method='L-BFGS-B',
            options={
                'maxiter': 1000,
                'disp': True
            })

    with tf.Session(graph=graph) as session:
        session.run(tf.global_variables_initializer())
        optimizer.minimize(session)
        waveform = session.run(noise)
    return waveform

for jj in range(n_plot):
    spectrogram = preds[:, jj] * train_itr._std + train_itr._mean
    reconstructed_waveform = sonify(spectrogram, len(spectrogram) * step, logmel)
    wavfile.write("sample_{}_pre.wav".format(jj), sample_rate, soundsc(reconstructed_waveform))

    fftsize = 512
    substep = 32
    rw_s = np.abs(stft(reconstructed_waveform, fftsize=fftsize, step=substep, real=False,
                       compute_onesided=False))
    rw = iterate_invert_spectrogram(rw_s, fftsize, substep, n_iter=100, verbose=True)
    wavfile.write("sample_{}_post.wav".format(jj), sample_rate, soundsc(rw))
