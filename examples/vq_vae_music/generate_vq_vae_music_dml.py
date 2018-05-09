import argparse
import tensorflow as tf
import numpy as np
from tfbldr.datasets import fetch_mnist
from collections import namedtuple
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

d = np.load("music_data.npz")
image_data = 2 * d["measures"] - 1.
# get images from the held out valid set
image_data = image_data[-1000:]

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
              'x_tilde_lin_scales',
              'x_tilde_coeffs']
    vs = namedtuple('Params', fields)(
        *[tf.get_collection(name)[0] for name in fields]
    )
    x = image_data[:64]
    feed = {vs.images: x,
            vs.bn_flag: 1.}
    outs = [vs.z_e_x, vs.z_q_x, vs.z_i_x, vs.x_tilde_mix, vs.x_tilde_means, vs.x_tilde_lin_scales, vs.x_tilde_coeffs]
    r = sess.run(outs, feed_dict=feed)
    logit_mixtures = r[-4]
    means = r[-3]
    lin_scales = r[-2]
    coeffs = r[-1]

    # gumbel sample to get mixture
    mixture_idx = np.argmax(logit_mixtures + -np.log(-np.log(random_state.uniform(1E-5, 1-1E-5, logit_mixtures.shape))), axis=-1)
    # components, make it one hot
    mixture_idx_oh = np.eye(10)[mixture_idx.ravel()].reshape(mixture_idx.shape + (-1,))

    shp = means.shape
    # channels, 10 components
    means = means.reshape(shp[:-1] + (3, 10))
    lin_scales = lin_scales.reshape(shp[:-1] + (3, 10))
    coeffs = np.tanh(coeffs.reshape(shp[:-1] + (3, 10)))

    mio = mixture_idx_oh[:, :, :, None, :]
    mmeans = (mio * means).sum(axis=-1)
    mlin_scales = (mio * lin_scales).sum(axis=-1)
    mcoeffs = (mio * coeffs).sum(axis=-1)

    u = random_state.uniform(1E-5, 1-1E-5, mmeans.shape)
    #  x = means + tf.exp(log_scales)*(tf.log(u) - tf.log(1. - u))
    #  x0 = tf.minimum(tf.maximum(x[:,:,:,0], -1.), 1.)
    #  x1 = tf.minimum(tf.maximum(x[:,:,:,1] + coeffs[:,:,:,0]*x0, -1.), 1.)
    #  x2 = tf.minimum(tf.maximum(x[:,:,:,2] + coeffs[:,:,:,1]*x0 + coeffs[:,:,:,2]*x1, -1.), 1.)
    #  return tf.concat(3,[tf.reshape(x0,xs[:-1]+[1]), tf.reshape(x1,xs[:-1]+[1]), tf.reshape(x2,xs[:-1]+[1])])
    xr = mmeans + np.exp(mlin_scales) * (np.log(u) - np.log(1. - u))

    def mima(a):
        return np.minimum(np.maximum(a, -1.), 1.)

    xr0 = mima(xr[..., 0])
    xr1 = mima(xr[..., 1] + mcoeffs[..., 0] * xr0)
    xr2 = mima(xr[..., 2] + mcoeffs[..., 1] * xr0 + mcoeffs[..., 2] * xr1)
    x_rec = np.concatenate((xr0[..., None], xr1[..., None], xr2[..., None]), axis=-1)
    x_rec = 0.5 * (x_rec + 1)
    x = 0.5 * (x + 1)
    x_rec[x_rec > 0.1] = 1.

    plt_height = 5
    plt_width = 5
    w = 48
    h = 48
    f, axarr = plt.subplots(1, 3)
    gt_storage = np.zeros((plt_height * h, plt_width * w, 3))
    rec_storage = np.zeros((plt_height * h, plt_width * w, 3))
    abs_rec_storage = np.zeros((plt_height * h, plt_width * w, 3))
    for i in range(plt_height):
        for j in range(plt_width):
            o_x_i = x[i * plt_width + j]
            r_x_i = x_rec[i * plt_width + j]
            gt_storage[h * i:h * (i + 1), w * j:w * (j + 1)] = o_x_i
            rec_storage[h * i:h * (i + 1), w * j:w * (j + 1)] = r_x_i 
            abs_rec_storage[h * i:h * (i + 1), w * j:w * (j + 1)] = np.abs(o_x_i - r_x_i)

    axarr[0].imshow(gt_storage, interpolation=None)
    axarr[0].set_title("Ground Truth")
    axarr[0].set_xticks([], [])
    axarr[0].set_yticks([], [])
    axarr[1].imshow(rec_storage, interpolation=None)
    axarr[1].set_title("Reconstruction")
    axarr[1].set_xticks([], [])
    axarr[1].set_yticks([], [])
    axarr[2].imshow(abs_rec_storage, interpolation=None)
    axarr[2].set_title("Absolute difference")
    axarr[2].set_xticks([], [])
    axarr[2].set_yticks([], [])
    plt.savefig("vq_vae_generation_results")
    plt.close()
