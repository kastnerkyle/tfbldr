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

d = np.load("music_data_jos.npz")
image_data = d["measures"]
image_data = np.concatenate((image_data[..., 0][..., None],
                             image_data[..., 1][..., None],
                             image_data[..., 2][..., None]), axis=0)
shuffle_random = np.random.RandomState(112)
shuffle_random.shuffle(image_data)
# get images from the held out valid set
image_data = image_data[-5000:]

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
    x = image_data[:64]
    feed = {vs.images: x,
            vs.bn_flag: 1.}
    outs = [vs.z_e_x, vs.z_q_x, vs.z_i_x, vs.x_tilde]
    r = sess.run(outs, feed_dict=feed)
    x_rec = r[-1]

    x_rec[x_rec > 0.1] = 1.
    x_rec[x_rec <= 0.1] = 0.

    plt_height = 5
    plt_width = 5
    w = 48
    h = 48
    f, axarr = plt.subplots(1, 3)
    gt_storage = np.zeros((plt_height * h, plt_width * w, 1))
    rec_storage = np.zeros((plt_height * h, plt_width * w, 1))
    abs_rec_storage = np.zeros((plt_height * h, plt_width * w, 1))
    for i in range(plt_height):
        for j in range(plt_width):
            o_x_i = x[i * plt_width + j]
            r_x_i = x_rec[i * plt_width + j]
            gt_storage[h * i:h * (i + 1), w * j:w * (j + 1)] = o_x_i
            rec_storage[h * i:h * (i + 1), w * j:w * (j + 1)] = r_x_i 
            abs_rec_storage[h * i:h * (i + 1), w * j:w * (j + 1)] = np.abs(o_x_i - r_x_i)

    axarr[0].imshow(gt_storage[:, :, 0], cmap="gray", interpolation=None)
    axarr[0].set_title("Ground Truth")
    axarr[0].set_xticks([], [])
    axarr[0].set_yticks([], [])
    axarr[1].imshow(rec_storage[:, :, 0], cmap="gray", interpolation=None)
    axarr[1].set_title("Reconstruction")
    axarr[1].set_xticks([], [])
    axarr[1].set_yticks([], [])
    axarr[2].imshow(abs_rec_storage[:, :, 0], cmap="gray", interpolation=None)
    axarr[2].set_title("Absolute difference")
    axarr[2].set_xticks([], [])
    axarr[2].set_yticks([], [])
    plt.savefig("vq_vae_generation_results")
    plt.close()
