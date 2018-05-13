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

d = np.load("music_data_jos.npz")
flat_images = np.array([mai for amai in copy.deepcopy(d['measures_as_images']) for mai in amai])
flat_idx = np.array([(i, j) for i, amai in enumerate(copy.deepcopy(d['measures_as_images']))
                            for j in range(len(amai))])
image_data = flat_images

bs = 50
image_data = image_data[:len(image_data) - len(image_data) % bs]
flat_idx = flat_idx[:len(flat_idx) - len(flat_idx) % bs]
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
    z = []
    for i in range(len(image_data) // bs):
        print("Minibatch {}".format(i))
        x = image_data[i * bs:(i + 1) * bs]
        feed = {vs.images: x,
                vs.bn_flag: 1.}

        outs = [vs.z_e_x, vs.z_q_x, vs.z_i_x, vs.x_tilde]
        r = sess.run(outs, feed_dict=feed)
        x_rec = r[-1]
        z_i = r[-2]
        z += [zz[:, :, None] for zz in z_i]
    z = np.array(z)

chordnames = d["chordnames"]
scalenotes = d["scalenotes"]
note_to_norm_kv = d["note_to_norm_kv"]
midi_to_norm_kv = d["midi_to_norm_kv"]

nested_chordnames = chordnames
flat_chordnames = [list(set(nested_chordnames[i][j]))[0] for i, j in flat_idx]
full_chord_lookup = {v: k for k, v in enumerate(list(sorted(set(flat_chordnames))))}
basic_chord_lookup = {v: k for k, v in enumerate(sorted(list(set(["".join([i for i in c if not i.isdigit()])
                                      for c in list(sorted(set(flat_chordnames)))]))))}

# iterate the chordnames and build the conditioning label
# label calc , 24 for left, 24 for right, 334 for cur 
chord_indices = []
for fi in flat_idx:
    cur_name = list(set(nested_chordnames[fi[0]][fi[1]]))[0] #assumes chord is the same over the measure...
    cur = full_chord_lookup[cur_name]
    if fi[1] == 0:
        left = 0
    else:
        left_name = list(set(nested_chordnames[fi[0]][fi[1] - 1]))[0]
        left = basic_chord_lookup["".join([i for i in left_name if not i.isdigit()])] + 1

    if fi[1] == (len(nested_chordnames[fi[0]]) - 1):
        right = 0
    else:
        right_name = list(set(nested_chordnames[fi[0]][fi[1] + 1]))[0]
        right = basic_chord_lookup["".join([i for i in right_name if not i.isdigit()])] + 1
    chord_indices.append((left, cur, right))

label_to_lcr = {}
lcr_to_label = {}
labels = []
counter = 0
for ci in chord_indices:
    if ci not in lcr_to_label:
        lcr_to_label[ci] = counter
        label_to_lcr[counter] = ci
        counter += 1
    lbl_i = lcr_to_label[ci]
    labels.append(lbl_i)

labels = np.array(labels)[:, None]
labels = [lbl_i for lbl_i in labels]
# savez is very upset if v is a tuple in label_to_lcr_kv, hackily convert to a string for now
# to convert
# ",".join([str(iv) for iv in (0, 252, 10)])
# to unconvert
# tuple([int(i) for i in '0,252,10'.split(",")])
# full lookup
# {k: tuple([int(iv) for iv in v.split(",")]) for k, v in label_to_lcr_kv}
label_to_lcr_kv = [(k, ",".join([str(iv) for iv in v])) for k, v in label_to_lcr.items()]
full_chords_kv = [(k, v) for k, v in full_chord_lookup.items()]
basic_chords_kv = [(k, v) for k, v in basic_chord_lookup.items()]

np.savez("vq_vae_encoded_music_2d.npz",
         z=z,
         labels=labels,
         flat_images=image_data,
         flat_idx=flat_idx,
         chordnames=chordnames,
         scalenotes=scalenotes,
         note_to_norm_kv=note_to_norm_kv,
         midi_to_norm_kv=midi_to_norm_kv,
         label_to_lcr_kv=label_to_lcr_kv,
         full_chords_kv=full_chords_kv,
         basic_chords_kv=basic_chords_kv
         )
print("dumped to vq_vae_encoded_music_2d.npz")
from IPython import embed; embed(); raise ValueError()
