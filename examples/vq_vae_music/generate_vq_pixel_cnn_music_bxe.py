import argparse
import tensorflow as tf
import numpy as np
from tfbldr.datasets import save_image_array
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
#mnist = fetch_mnist()
image_data = d1["measures"]
image_data = np.concatenate((image_data[..., 0][..., None],
                             image_data[..., 1][..., None],
                             image_data[..., 2][..., None]), axis=0)
# times 0 to ensure NO information leakage
valid_image_data = 0. * image_data 
d2 = np.load("vq_vae_encoded_music.npz")
valid_image_labels = d2["valid_labels"]


random_state = np.random.RandomState(2000)
def sample_gumbel(logits, temperature=1.):
    noise = random_state.uniform(1E-5, 1. - 1E-5, np.shape(logits))
    #return np.argmax(np.log(softmax(logits, temperature)) - np.log(-np.log(noise)))
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
    y = valid_image_labels[:num_to_generate]

    pix_z = np.zeros((num_to_generate, 3, 3))
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
    x = valid_image_data[:num_to_generate]
    z_i = pix_z[:num_to_generate]
    # again multiply by 0 to avoid information leakage
    feed = {vs.images: 0. * x,
            vs.z_i_x: z_i,
            vs.bn_flag: 1.}
    outs = [vs.x_tilde]
    r = sess2.run(outs, feed_dict=feed)
    x_rec = r[-1]

x_rec[x_rec < 0.3] = 0.
x_rec[x_rec > 0.3] = 1.
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
