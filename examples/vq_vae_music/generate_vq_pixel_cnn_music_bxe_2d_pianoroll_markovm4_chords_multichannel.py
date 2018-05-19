import argparse
import tensorflow as tf
import numpy as np
from tfbldr.datasets import piano_roll_imlike_to_image_array
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

from decode import decode_measure

parser = argparse.ArgumentParser()
parser.add_argument('pixelcnn_model', nargs=1, default=None)
parser.add_argument('vqvae_model', nargs=1, default=None)
parser.add_argument('--seed', dest='seed', type=int, default=1999)
parser.add_argument('--temp', dest='temp', type=float, default=1.)
args = parser.parse_args()
vqvae_model_path = args.vqvae_model[0]
pixelcnn_model_path = args.pixelcnn_model[0]

num_to_generate = 1000
num_each = 64
random_state = np.random.RandomState(args.seed)

d1 = np.load("music_data_jos_pianoroll_multichannel.npz")
flat_images = np.array([mai for amai in copy.deepcopy(d1['measures_as_images']) for mai in amai])
image_data = flat_images

# times 0 to ensure NO information leakage
sample_image_data = 0. * image_data
# shuffle it just because
random_state.shuffle(sample_image_data)

d2 = np.load("vq_vae_encoded_music_jos_2d_pianoroll_multichannel.npz")

# use these to generate
flat_idx = d2["flat_idx"]
sample_flat_idx = flat_idx[-1000:]
labels = d2["labels"]
sample_labels = labels[-1000:]

def sample_gumbel(logits, temperature=args.temp):
    noise = random_state.uniform(1E-5, 1. - 1E-5, np.shape(logits))
    return np.argmax((logits - np.max(logits) - 1) / float(temperature) - np.log(-np.log(noise)), axis=-1)

config = tf.ConfigProto(
    device_count={'GPU': 0}
)

with tf.Session(config=config) as sess1:
    saver = tf.train.import_meta_graph(pixelcnn_model_path + '.meta')
    saver.restore(sess1, pixelcnn_model_path)
    fields = ['images',
              'conds',
              'labels',
              'x_tilde']
    vs = namedtuple('Params', fields)(
        *[tf.get_collection(name)[0] for name in fields]
    )

    y = sample_labels[:num_to_generate]
    sampled_z = []
    c = np.zeros((1, 12, 6))
    for sample_step in range(num_to_generate):
        print("step {}".format(sample_step))
        pix_z = np.zeros((1, 12, 6))
        if sample_flat_idx[sample_step][1] < 4:
            # reset it on song boundaries - will be used again when combined labels
            c = c * 0
        for i in range(pix_z.shape[1]):
            for j in range(pix_z.shape[2]):
                print("Sampling v completion pixel {}, {}".format(i, j))
                feed = {vs.images: pix_z[..., None],
                        vs.conds: c[..., None],
                        vs.labels: y[sample_step][None]}
                outs = [vs.x_tilde]
                r = sess1.run(outs, feed_dict=feed)
                x_rec = sample_gumbel(r[-1])

                for k in range(pix_z.shape[0]):
                    pix_z[k, i, j] = float(x_rec[k, i, j])
        sampled_z.append(pix_z)
        if sample_step >= 4:
            c = sampled_z[-4]

sess1.close()
tf.reset_default_graph()

sampled_z = np.array(sampled_z)
pix_z = sampled_z[:, 0]

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
    x = image_data[:num_to_generate]
    z_i = pix_z[:num_to_generate]
    # again multiply by 0 to avoid information leakage
    feed = {vs.images: 0. * x,
            vs.z_i_x: z_i,
            vs.bn_flag: 1.}
    outs = [vs.x_tilde]
    r = sess2.run(outs, feed_dict=feed)
    x_rec = r[-1]

# binarize the predictions
x_rec[x_rec > 0.5] = 1.
x_rec[x_rec <= 0.5] = 0.

full_chords_kv = d2["full_chords_kv"]
label_to_lcr_kv = d2["label_to_lcr_kv"]
basic_chords_kv = d2["basic_chords_kv"]
full_chords_kv = d2["full_chords_kv"]

label_to_lcr = {int(k): tuple([int(iv) for iv in v.split(",")]) for k, v in label_to_lcr_kv}
full_chords_lu = {k: int(v) for k, v in full_chords_kv}
basic_chords_lu = {k: int(v) for k, v in full_chords_kv}

"""
# find some start points
for n in range(len(sample_labels)):
    lcr_i = label_to_lcr[sample_labels[n, 0]]
    if lcr_i[0] == 0:
        print(n) 
"""
# 16 44 117 119 143 151 206 242 267 290 308 354 380 410 421 456 517 573 598 622 638 663 676 688 715 725 749 752 820 851 866 922

# start at 16 since that's the start of a chord sequence (could choose any of the numbers above)
for offset in [16, 44, 308, 421, 517, 752, 866]:
    print("sampling offset {}".format(offset))
    x_rec_i = x_rec[offset:offset + num_each]

    x_ts = piano_roll_imlike_to_image_array(x_rec_i, 0.25)

    if not os.path.exists("samples"):
        os.mkdir("samples")
    save_image_array(x_ts, "samples/pianoroll_multichannel_pixel_cnn_markovm4_chords_gen_{}_seed_{}_temp_{}.png".format(offset, args.seed, args.temp))

    sample_flat_idx = flat_idx[-1000:]

    p = sample_flat_idx[offset:offset + num_each]

    satb_midi = [[], [], [], []]
    satb_notes = [[], [], [], []]
    for n in range(len(x_rec_i)):
        measure_len = x_rec_i[n].shape[1]
        # 96 x 48 measure in
        events = {}
        for v in range(x_rec_i.shape[-1]):
            all_up = zip(*np.where(x_rec_i[n][..., v]))
            time_ordered = [au for i in range(measure_len) for au in all_up if au[1] == i]
            for to in time_ordered:
                if to[1] not in events:
                    # fill with rests
                    events[to[1]] = [0, 0, 0, 0]
                events[to[1]][v] = to[0]

        satb =[[], [], [], []]
        for v in range(x_rec_i.shape[-1]):
            for ts in range(measure_len):
                if ts in events:
                    satb[v].append(events[ts][v])
                else:
                    # edge case if ALL voices rest
                    satb[v].append(0)

        # was ordered btas
        satb = satb[::-1]
        for i in range(len(satb)):
            satb_midi[i].extend(satb[i])
            satb_notes[i].extend(midi_to_notes([satb[i]])[0])

    quantized_to_pretty_midi([satb_midi],
                             0.25,
                             save_dir="samples",
                             name_tag="pianoroll_multichannel_markovm4_chords_sample_{}_seed_{}_temp_{}".format(offset, args.seed, args.temp) + "_{}.mid",
                             default_quarter_length=220,
                             voice_params="woodwinds")
    print("saved sample {}".format(offset))
from IPython import embed; embed(); raise ValueError()
