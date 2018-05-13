import argparse
import tensorflow as tf
import numpy as np
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


parser = argparse.ArgumentParser()
parser.add_argument('pixelcnn_model', nargs=1, default=None)
parser.add_argument('vqvae_model', nargs=1, default=None)
parser.add_argument('--seed', dest='seed', type=int, default=1999)
args = parser.parse_args()
vqvae_model_path = args.vqvae_model[0]
pixelcnn_model_path = args.pixelcnn_model[0]

num_to_generate = 256
num_to_plot = 16
random_state = np.random.RandomState(args.seed)

d1 = np.load("music_data_jos.npz")
flat_images = np.array([mai for amai in copy.deepcopy(d1['measures_as_images']) for mai in amai])
image_data = flat_images

# times 0 to ensure NO information leakage
sample_image_data = 0. * image_data 
d2 = np.load("vq_vae_encoded_music_jos_2d.npz")

# use these to generate
labels = d2["labels"]
sample_labels = labels[-1000:]

random_state = np.random.RandomState(2000)
def sample_gumbel(logits, temperature=1.):
    noise = random_state.uniform(1E-5, 1. - 1E-5, np.shape(logits))
    return np.argmax(logits / float(temperature) - np.log(-np.log(noise)), axis=-1)

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

    pix_z = np.zeros((num_to_generate, 6, 6))
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
# many of these need to choose
scalenotes = d2["scalenotes"]

label_to_lcr = {int(k): tuple([int(iv) for iv in v.split(",")]) for k, v in label_to_lcr_kv}
full_chords_lu = {k: int(v) for k, v in full_chords_kv}
basic_chords_lu = {k: int(v) for k, v in full_chords_kv}

flat_scalenotes = [sn for sg in copy.deepcopy(d2["scalenotes"]) for sn in sg]
sample_scalenotes = flat_scalenotes[-1000:]

"""
for n in range(len(sample_labels)):
    lcr_i = label_to_lcr[sample_labels[n, 0]]
    if lcr_i[0] == 0:
        print(n) 
"""
# 16 44 117 119 143 151 206 242 267 290 308 354 380 410 421 456 517 573 598 622 638 663 676 688 715 725 749 752 820 851 866 922

# start at 16 since that's the start of a chord sequence (could choose any of the numbers above)
offset = 16
x_rec_i = x_rec[offset:offset + num_to_plot]
save_image_array(x_rec_i, "pixel_cnn_gen.png")

used_scale = sample_scalenotes[offset]
note_to_norm_lu = {}
note_to_norm_lu["R"] = 0
midi_to_norm_lu = {}
midi_to_norm_lu[0] = 0
counter = 1
for octave in ["1", "2", "3", "4", "5"]:
    for note in used_scale:
        nnm = notes_to_midi([[note + octave]])[0][0]
        nn = midi_to_notes([[nnm]])[0][0][:-1]
        note_to_norm_lu[note + octave] = counter
        midi_to_norm_lu[nnm] = counter
        counter += 1

norm_to_note_lu = {v: k for k, v in note_to_norm_lu.items()}
norm_to_midi_lu = {v: k for k, v in midi_to_norm_lu.items()}

norm_voices = []
for n in range(num_to_plot):
    measure = x_rec[offset + n, :, :, 0]
    all_up = list(zip(*np.where(measure)))
    time_ordered = [au for i in range(x_rec.shape[1]) for au in all_up if au[1] == i]
    events = {}
    for to in time_ordered:
        if to[1] not in events:
            events[to[1]] = []
        events[to[1]].append(to[0])

    last_non_rest = [0] * 4
    overall_seq = []
    for i in range(x_rec.shape[1]):
        ts = events[i]
        if len(ts) == 4:
        else:
            # figure out voice assignment based on history
            raise ValueError("NEED TO RETRAIN T_T")
            from IPython import embed; embed(); raise ValueError()
            for v in range(4):
                pass
        for v in range(4):
            if last_non_rest[v] == 0:
                last_non_rest[v] = ts[v]
            
    from IPython import embed; embed(); raise ValueError()

# smoothness
delta_counts = [np.sum(np.abs(np.diff(np.where(x_rec[i][:, :, 0])[0]))) for i in range(len(x_rec))]
simul = [np.max(np.sum(x_rec[i][:, :, 0], axis=0)) for i in range(len(x_rec))]
non_boring_indices = [i for i in range(len(x_rec)) if 0 < delta_counts[i] <= 12 and simul[i] <= 1] # includes rest measures, rests arent boring    

x_rec = x_rec[np.array(non_boring_indices)]
if len(x_rec) < (3 * num_to_plot):
    raise ValueError("Removed too many boring ones, set num_to_plot lower")

x_rec = x_rec[:3 * num_to_plot]
x_rec = np.concatenate((x_rec[::3], x_rec[1::3], x_rec[2::3]), axis=-1)

save_image_array(x_rec, "samp_vq_pixel_cnn_music_bxe.png")

ce = copy.deepcopy(d1["centers"])
# only use bottom 3 voices
ce = [cei for cei in ce if len(cei) == 4]
# find all the ones with 0 or 1 rest
non_rest = [i for i in range(len(ce)) if sum(ce[i] == 0) == 0]
start_chunks = [i for i in range(len(non_rest) - num_to_plot) if np.max(np.diff(non_rest[i:i+num_to_plot])) == 1]
random_state.shuffle(start_chunks)
random_state.shuffle(start_chunks)
ii = non_rest[start_chunks[0]]
skeleton = np.array([sk for sk in ce[ii:(ii + num_to_plot)]])
joined = np.zeros((len(x_rec), x_rec.shape[2], skeleton.shape[-1]))
joined += skeleton[:, None, :]
idxs = np.argmax(x_rec, axis=1)
lu = {k: v for k, v in enumerate(np.arange(-23, 24))}
res = np.zeros_like(idxs)
for kk in sorted(lu.keys()):
    res[idxs == kk] = lu[kk]

# use only top
res[:, :, 1:] *= 0
joined[:, :, :res.shape[-1]] += res

joined = joined.reshape(-1, joined.shape[-1])
joined_voices = [joined[:, c] for c in range(joined.shape[-1])]

quantized_to_pretty_midi([joined_voices],
                         0.25,
                         default_quarter_length=440,
                         voice_params="piano")
from IPython import embed; embed(); raise ValueError()
