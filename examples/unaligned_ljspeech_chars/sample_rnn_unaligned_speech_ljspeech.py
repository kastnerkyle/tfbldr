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
from tfbldr.datasets import rsync_fetch, fetch_ljspeech
from tfbldr.datasets import wavfile_caching_mel_tbptt_iterator
from tfbldr.datasets.text import pronounce_chars
from scipy.io import wavfile
from tfbldr.datasets.audio import soundsc
from tfbldr.datasets.audio import stft
from tfbldr.datasets.audio import iterate_invert_spectrogram


parser = argparse.ArgumentParser()
parser.add_argument('direct_model', nargs=1, default=None)
parser.add_argument('--model', dest='model_path', type=str, default=None)
parser.add_argument('--seed', dest='seed', type=int, default=1999)
parser.add_argument('--test', dest='test', type=str, default="valid")
parser.add_argument('--inp', dest='input_type', type=str, default="blend++")
parser.add_argument('--bs', dest='batch_size', type=int, default=64)
parser.add_argument('--sonify', dest='sonify', type=int, default=100)
parser.add_argument('--gl', dest='gl', type=int, default=100)
parser.add_argument('--chars', dest='chars', action="store_true")
parser.add_argument('--phones', dest='phones', action="store_true")
parser.add_argument('--both', dest='both', action="store_true")
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
ljspeech = rsync_fetch(fetch_ljspeech, "leto01")
#ljspeech = fetch_ljspeech()
wavfiles = ljspeech["wavfiles"]
jsonfiles = ljspeech["jsonfiles"]

batch_size = args.batch_size
seq_len = 256
window_mixtures = 10
enc_units = 128
dec_units = 512
emb_dim = 15
sonify_steps = args.sonify
gl_steps = args.gl

sample_rate = 22050
window_size = 512
step = 128
n_mel = 80

itr_random_state = np.random.RandomState(3122)

has_mask = True
if args.chars or args.input_type == "chars":
    valid_itr = wavfile_caching_mel_tbptt_iterator(wavfiles, jsonfiles, batch_size, seq_len, symbol_processing="chars_only", start_index=.95, shuffle=True, random_state=itr_random_state)
    args.input_type = "chars"
elif args.phones or args.input_type == "phones":
    valid_itr = wavfile_caching_mel_tbptt_iterator(wavfiles, jsonfiles, batch_size, seq_len, symbol_processing="phones_only", start_index=.95, shuffle=True, random_state=itr_random_state)
    args.input_type = "phones"
elif args.input_type == "blend++":
    valid_itr = wavfile_caching_mel_tbptt_iterator(wavfiles, jsonfiles, batch_size, seq_len, symbol_processing="blended_pref", start_index=.95, shuffle=True, random_state=itr_random_state)
elif args.input_type == "blend":
    valid_itr = wavfile_caching_mel_tbptt_iterator(wavfiles, jsonfiles, batch_size, seq_len, symbol_processing="blended", start_index=.95, shuffle=True, random_state=itr_random_state)

if args.chars or args.phones:
    has_mask = False
itr = valid_itr
mels, mel_mask, text, text_mask, mask, mask_mask, reset = itr.next_masked_batch()
if args.test == "valid":
    n_to_sample = 4
elif args.test in ["basic", "quote", "full", "taco_prosody", "taco_small", "custom"]:
    with open("{}_test.txt".format(args.test)) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    n_to_sample = len(lines)
    int_lines = []
    print("lines to sample:")
    masks = []
    for l in lines:
        tt, mm = valid_itr.transform_txt(l)
        int_lines.append(tt)
        masks.append(mm)
        ot = valid_itr.inverse_transform_txt(tt, mm)
        print(ot)
    longest = max([len(il) for il in int_lines])
    text = np.zeros((longest, batch_size, 1))
    text_mask = np.zeros((longest, batch_size))
    mask = np.zeros((longest, batch_size, 1))
    mask_mask = np.zeros((longest, batch_size))
    for n, il in enumerate(int_lines):
        text[:len(il), n, 0] = il
        text_mask[:len(il), n] = 1.
        mask[:len(il), n, 0] = masks[n]
        mask_mask[:len(il), n] = 1.
else:
    raise ValueError("Invalid argument for args.test")

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
              #"prenet_dropout",
              #"attskip_dropout",
              "bn_flag",
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
    if has_mask:
        fields += ["mask", "mask_mask"]
    vs = namedtuple('Params', fields)(
        *[tf.get_collection(name)[0] for name in fields]
    )
    att_w_init = np.zeros((batch_size, 2 * enc_units))
    att_k_init = np.zeros((batch_size, window_mixtures))
    att_h_init = np.zeros((batch_size, dec_units))
    att_c_init = np.zeros((batch_size, dec_units))
    h1_init = np.zeros((batch_size, dec_units))
    c1_init = np.zeros((batch_size, dec_units))
    h2_init = np.zeros((batch_size, dec_units))
    c2_init = np.zeros((batch_size, dec_units))

    """
    # only predict 1 thing
    endpoint = np.where(text_mask[:, 0] == 1)[0][-1]
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
    #noise_scale = .5
    is_finished_sampling = [False] * n_to_sample
    finished_at = 100000000
    finished_step = [-1] * n_to_sample

    ii = 0

    # add ~.1 sec to the end
    # sample_rate * .1 / fft_step
    # min_part to account for the last window
    min_part = window_size / float(sample_rate)
    extra_steps = max(0, int((sample_rate * (.1 - min_part)) / float(step)))
    while True:
        print("pred step {}".format(ii))
        #noise_block = np.clip(random_state.randn(*in_mels.shape), -6, 6)
        #in_mels = in_mels + noise_scale * noise_block
        #if ii > 0:
        #    in_mels[0] = mels[ii - 1]
        feed = {
                vs.in_mels: in_mels,
                vs.in_mel_mask: in_mel_mask,
                vs.bn_flag: 1.,
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
        if has_mask:
            feed[vs.mask] = mask
            feed[vs.mask_mask] = mask_mask
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

        ii += 1
        max_text = max([text_mask[:, mbi].sum() for mbi in range(n_to_sample)])
        if ii > 30 * max_text:
            # it's gone too far, kill
            finished_step = [int(30 * max_text)] * n_to_sample
            print("Exceeded 30 * max text length of {},  terminating...".format(max_text))
            break

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

        for mbi in range(n_to_sample):
            last_sym = int(text_mask[:, mbi].sum()) - 1
            if np.argmax(att_phi_np[0, mbi]) >= last_sym or np.argmax(att_phi_np[0, mbi]) == text_mask.shape[0]:
                if is_finished_sampling[mbi] == False:
                    is_finished_sampling[mbi] = True
                    finished_step[mbi] = ii

        if all(is_finished_sampling):
            print("All samples finished at step {}".format(finished_at))
        else:
            # should assign until all are finished
            finished_at = ii

        if ii > (finished_at + extra_steps):
            print("Extra padding {} finished at step {}".format(extra_steps, ii))
            break

preds = np.array(preds)
n_plot = n_to_sample
f, axarr = plt.subplots(1, n_plot)
for jj in range(n_plot):
    spectrogram = preds[:, jj] * itr._std + itr._mean
    axarr[jj].imshow(spectrogram)
plt.savefig("{}_{}_sample_spec.png".format(args.test, args.input_type))

att_phis = np.array(att_phis)
f, axarr = plt.subplots(1, n_plot)
for jj in range(n_plot):
    phi_i = att_phis[:, jj]
    axarr[jj].imshow(phi_i)
plt.savefig("{}_{}_sample_att.png".format(args.test, args.input_type))

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

        # tf.nn.normalize arguments changed between versions...
        def normalize(a):
            return a / tf.sqrt(tf.maximum(tf.reduce_sum(a ** 2, axis=0), 1E-12))

        x = normalize(x)
        y = normalize(y)
        tf.losses.mean_squared_error(x, y[-tf.shape(x)[0]:])

        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            loss=tf.losses.get_total_loss(),
            var_list=[noise],
            tol=1e-16,
            method='L-BFGS-B',
            options={
                'maxiter': sonify_steps,
                'disp': True
            })

    with tf.Session(graph=graph) as session:
        session.run(tf.global_variables_initializer())
        optimizer.minimize(session)
        waveform = session.run(noise)
    return waveform

for jj in range(n_plot):
    # use extra steps from earlier
    pjj = preds[:(finished_step[jj] + extra_steps), jj]
    spectrogram = pjj * itr._std + itr._mean
    reconstructed_waveform = sonify(spectrogram, len(spectrogram) * step, logmel)
    wavfile.write("{}_{}_sample_{}_pre.wav".format(args.test, args.input_type, jj), sample_rate, soundsc(reconstructed_waveform))

    fftsize = 512
    substep = 32
    rw_s = np.abs(stft(reconstructed_waveform, fftsize=fftsize, step=substep, real=False,
                       compute_onesided=False))
    rw = iterate_invert_spectrogram(rw_s, fftsize, substep, n_iter=gl_steps, verbose=True)
    wavfile.write("{}_{}_sample_{}_post.wav".format(args.test, args.input_type, jj), sample_rate, soundsc(rw))
