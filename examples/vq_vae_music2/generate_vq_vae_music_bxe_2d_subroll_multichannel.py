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
from tfbldr.datasets import fetch_josquin
from tfbldr.datasets import quantized_imlike_to_image_array
from data_utils import music_pitch_and_chord_to_imagelike_and_label


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

josquin = fetch_josquin()
images, labels, lookups = music_pitch_and_chord_to_imagelike_and_label(josquin)

image_data = images
image_data = image_data[-500:]
#shuffle_random = np.random.RandomState(112)
#shuffle_random.shuffle(image_data)
# get images from the held out valid set

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
    x = image_data[:32]
    feed = {vs.images: x,
            vs.bn_flag: 1.}
    outs = [vs.z_e_x, vs.z_q_x, vs.z_i_x, vs.x_tilde]
    r = sess.run(outs, feed_dict=feed)
    x_rec = r[-1]

    x_rec[x_rec > 0.5] = 1.
    x_rec[x_rec <= 0.5] = 0.

    #from IPython import embed; embed(); raise ValueError()
    rr = quantized_imlike_to_image_array(image_data[:16], 0.25)
    save_image_array(rr, "orig_subroll_multichannel.png", resize_multiplier=(4, 1), gamma_multiplier=7, flat_wide=True)

    rr = quantized_imlike_to_image_array(x_rec[:16], 0.25)
    save_image_array(rr, "rec_subroll_multichannel.png", resize_multiplier=(4, 1), gamma_multiplier=7, flat_wide=True)
    print("wrote out 'orig_subroll_multichannel.png'")
    print("wrote out 'rec_subroll_multichannel.png'")
