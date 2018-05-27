import argparse
import tensorflow as tf
import numpy as np
from tfbldr.datasets import quantized_imlike_to_image_array
from tfbldr.datasets import save_image_array
from tfbldr.datasets import notes_to_midi
from tfbldr.datasets import midi_to_notes
from collections import namedtuple
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy
from tfbldr.datasets import quantized_to_pretty_midi
import os

from data_utils import dump_subroll_samples

parser = argparse.ArgumentParser()
parser.add_argument('pixelcnn_model', nargs=1, default=None)
parser.add_argument('vqvae_model', nargs=1, default=None)
parser.add_argument('--seed', dest='seed', type=int, default=1999)
parser.add_argument('--temp', dest='temp', type=float, default=1.)
parser.add_argument('--chords', dest='chords', type=str, default=None)
args = parser.parse_args()
vqvae_model_path = args.vqvae_model[0]
pixelcnn_model_path = args.pixelcnn_model[0]

num_to_generate = 160
num_each = 16
random_state = np.random.RandomState(args.seed)

d = np.load("vq_vae_encoded_music_2d_subroll_multichannel.npz")

offset_to_pitch = {int(k): int(v) for k, v in d["offset_to_pitch_kv"]}
label_to_chord_function = {int(k): v for k, v in d["label_to_chord_function_kv"]}

labels = d["labels"]
train_labels = labels[:-500]
valid_labels = labels[-500:]
sample_labels = valid_labels

"""
if args.chords != None:
    raise ValueError("")
    chordseq = args.chords.split(",")
    if len(chordseq) < 3:
        raise ValueError("Provided chords length < 3, need at least 3 chords separated by spaces! Example: --chords=I7,IV7,V7,I7 . Got {}".format(args.chords))
    ch =  [cs for cs in args.chords.split(",")]
    clbl = [full_chords_lu[cs] for cs in ch]
    stretched = clbl * (num_to_generate // len(clbl) + 1)
    sample_labels = np.array(stretched[:len(sample_labels)])[:, None]
"""

def sample_gumbel(logits, temperature=args.temp):
    noise = random_state.uniform(1E-5, 1. - 1E-5, np.shape(logits))
    return np.argmax((logits - logits.max() - 1) / float(temperature) - np.log(-np.log(noise)), axis=-1)

config = tf.ConfigProto(
    device_count={'GPU': 0}
)

with tf.Session(config=config) as sess1:
    saver = tf.train.import_meta_graph(pixelcnn_model_path + '.meta')
    saver.restore(sess1, pixelcnn_model_path)
    fields = ['images',
              'labels',
              'x_tilde']
    vs = namedtuple('Params', fields)(
        *[tf.get_collection(name)[0] for name in fields]
    )
    y = sample_labels[:num_to_generate]

    pix_z = np.zeros((num_to_generate, 12, 8))
    for i in range(pix_z.shape[1]):
        for j in range(pix_z.shape[2]):
            print("Sampling v completion pixel {}, {}".format(i, j))
            feed = {vs.images: pix_z[..., None],
                    vs.labels: y}
            outs = [vs.x_tilde]
            r = sess1.run(outs, feed_dict=feed)
            x_rec = sample_gumbel(r[-1])

            for k in range(pix_z.shape[0]):
                pix_z[k, i, j] = float(x_rec[k, i, j])


sess1.close()
tf.reset_default_graph()

with tf.Session(config=config) as sess2:
    saver = tf.train.import_meta_graph(vqvae_model_path + '.meta')
    saver.restore(sess2, vqvae_model_path)
    """
    # test by faking like we sampled these from pixelcnn
    d = np.load("vq_vae_encoded_mnist.npz")
    valid_z_i = d["valid_z_i"]
    """
    fields = ['images',
              'bn_flag',
              'z_e_x',
              'z_q_x',
              'z_i_x',
              'x_tilde']
    vs = namedtuple('Params', fields)(
        *[tf.get_collection(name)[0] for name in fields]
    )
    z_i = pix_z[:num_to_generate]
    fake_image_data = np.zeros((num_to_generate, 48, 32, 4))
    feed = {vs.images: fake_image_data,
            vs.z_i_x: z_i,
            vs.bn_flag: 1.}
    outs = [vs.x_tilde]
    r = sess2.run(outs, feed_dict=feed)
    x_rec = r[-1]

# binarize the predictions
x_rec[x_rec > 0.5] = 1.
x_rec[x_rec <= 0.5] = 0.

dump_subroll_samples(x_rec, sample_labels, num_each, args.seed, args.temp, offset_to_pitch, label_to_chord_function)
from IPython import embed; embed(); raise ValueError()
