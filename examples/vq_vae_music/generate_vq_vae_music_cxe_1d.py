import argparse
import tensorflow as tf
import numpy as np
from tfbldr.datasets import plot_piano_roll
from tfbldr.datasets import save_image_array
from collections import namedtuple
import sys
import matplotlib
import copy
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

d = np.load("music_data_1d.npz")
raw_data = copy.deepcopy(d["absolutes"])
stacked = [rdi[None] for rd in raw_data for rdi in rd]
# vocabulary
# np.vstack(sorted(list({tuple(row) for row in np.array(stacked).reshape(-1, 4)})))

image_data = stacked
shuffle_random = np.random.RandomState(112)
shuffle_random.shuffle(image_data)
# save last 1k to validate on
train_image_data = image_data[:-1000]
val_image_data = image_data[-1000:]
image_data = val_image_data

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
    x_tilde_lins = r[-1]

    # gumbel sample to get mixture
    idx = np.argmax(x_tilde_lins + -np.log(-np.log(random_state.uniform(1E-5, 1-1E-5, x_tilde_lins.shape))), axis=-1)
    x_rec = idx
    """
    # components, make it one hot
    idx_oh = np.eye(x_tilde_lins.shape[-1])[idx.ravel()].reshape(idx.shape + (-1,))
    # get rid of the useless channel
    idx_oh = idx_oh[:, 0]
    # transpose to h, w, c, time on horizontal
    x_rec = idx_oh.transpose(0, 3, 1, 2)
    """
    x = np.array(x).astype("int32")

    x_rec = x_rec[:4]
    x = x[:4]

    x_img = [plot_piano_roll(x[i][0], 0.25) for i in range(len(x))] 
    x_rec_img = [plot_piano_roll(x_rec[i][0], 0.25) for i in range(len(x_rec))] 

    save_image_array(np.array(x_img)[0][None], "original_1d.png", rescale=False)
    save_image_array(np.array(x_rec_img)[0][None], "reconstructed_1d.png", rescale=False)

    # convert back into real pitches?
    # jk its bad tho
