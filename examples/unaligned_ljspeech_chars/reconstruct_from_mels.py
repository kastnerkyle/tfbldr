from tfbldr.datasets.audio import fetch_sample_speech_tapestry
from tfbldr.datasets.audio import soundsc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import numpy as np
from scipy.io import wavfile
from tfbldr.datasets.audio import linear_to_mel_weight_matrix
from tfbldr.datasets.audio import stft
from tfbldr.datasets.audio import iterate_invert_spectrogram
from tfbldr.datasets import wavfile_caching_mel_tbptt_iterator
from tfbldr.datasets import fetch_ljspeech


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


ljspeech = fetch_ljspeech()
wavfiles = ljspeech["wavfiles"]
txtfiles = ljspeech["txtfiles"]

batch_size = 64
seq_len = 256
train_random_state = np.random.RandomState(3122)
train_itr = wavfile_caching_mel_tbptt_iterator(wavfiles, txtfiles, batch_size, seq_len, stop_index=.95, shuffle=True, random_state=train_random_state)
mels, mel_mask, text, text_mask, reset = train_itr.next_masked_batch()
#fs, d = fetch_sample_speech_tapestry()
orig_wavfile = train_itr.wavfile_list[train_itr.current_indices_[0]]
fs, d = wavfile.read(orig_wavfile)

spectrogram2 = mels * train_itr._std + train_itr._mean
spectrogram2 = spectrogram2[:, 0]


sample_rate = fs
window_size = 512
step = 128
n_mel = 80
wav_scale = 2 ** 15
waveform = d / float(wav_scale)
waveform = waveform.astype("float32")

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


def logmel2(waveform):
    res = np.abs(stft(waveform, windowsize=window_size, step=step, real=False, compute_onesided=True))
    mels = linear_to_mel_weight_matrix(
        res.shape[1],
        sample_rate,
        lower_edge_hertz=125.,
        upper_edge_hertz=7800.,
        n_filts=n_mel, dtype=np.float64)
    mel_res = np.dot(res, mels)
    return np.log1p(mel_res)

with tf.Session():
    #spectrogram = logmel(waveform).eval()
    spectrogram = logmel2(waveform)
    spectrogram = spectrogram[:len(spectrogram2)]
spectrogram = (spectrogram - spectrogram.min()) / float(spectrogram.max() - spectrogram.min())
spectrogram2 = (spectrogram2 - spectrogram2.min()) / float(spectrogram2.max() - spectrogram2.min())
#spectrogram2 = info["log_mels"].astype("float32")
#spectrogram2 = logmel2(waveform)
#spectrogram2 = (spectrogram2 - spectrogram2.min()) / float(spectrogram2.max() - spectrogram2.min())


f, axarr = plt.subplots(1, 2)
axarr[0].imshow(spectrogram)
axarr[1].imshow(spectrogram2)
plt.savefig("tmpspec")

reconstructed_waveform = sonify(spectrogram, len(spectrogram) * step, logmel)
#reconstructed_waveform = sonify(spectrogram, len(waveform), logmel)
wavfile.write("tmp.wav", sample_rate, soundsc(reconstructed_waveform))
reconstructed_waveform2 = sonify(spectrogram2, len(spectrogram2) * step, logmel)
#reconstructed_waveform2 = sonify(spectrogram2, len(waveform), logmel)
wavfile.write("tmp2.wav", sample_rate, soundsc(reconstructed_waveform2))


fftsize = 512
substep = 32
rw_s = np.abs(stft(reconstructed_waveform, fftsize=fftsize, step=substep, real=False,
                   compute_onesided=False))
rw = iterate_invert_spectrogram(rw_s, fftsize, substep, n_iter=100, verbose=True)

rw2_s = np.abs(stft(reconstructed_waveform2, fftsize=fftsize, step=substep, real=False,
                   compute_onesided=False))
rw2 = iterate_invert_spectrogram(rw2_s, fftsize, substep, n_iter=100, verbose=True)

d_s = np.abs(stft(waveform, fftsize=fftsize, step=substep, real=False,
                  compute_onesided=False))
df = iterate_invert_spectrogram(d_s, fftsize, substep, n_iter=10, verbose=True)
wavfile.write("tmpif.wav", sample_rate, soundsc(df))
wavfile.write("tmpf.wav", sample_rate, soundsc(rw))
wavfile.write("tmpf2.wav", sample_rate, soundsc(rw2))
