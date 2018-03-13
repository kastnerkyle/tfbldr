import argparse
import tensorflow as tf
import numpy as np
from tfbldr.datasets import fetch_mnist
from collections import namedtuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--model', dest='model_path', type=str, default=None)
parser.add_argument('--seed', dest='seed', type=int, default=1999)
args = parser.parse_args()
if args.model_path == None:
    raise ValueError("Must pass --model argument, e.g. summary/experiment-0/models/model-7")

random_state = np.random.RandomState(args.seed)

config = tf.ConfigProto(
    device_count={'GPU': 0}
)

mnist = fetch_mnist()
image_data = mnist["images"] / 255.
# get images from the held out valid set
image_data = image_data[-10000:]

with tf.Session(config=config) as sess:
    saver = tf.train.import_meta_graph(args.model_path + '.meta')
    saver.restore(sess, args.model_path)
    fields = ['images',
              'bn_flag',
              'z_e_x',
              'z_q_x',
              'z_i_x',
              'x_tilde']
    vs = namedtuple('Params', fields)(
        *[tf.get_collection(name)[0] for name in fields]
    )
    x = image_data[:10]
    feed = {vs.images: x,
            vs.bn_flag: 1.}
    outs = [vs.z_e_x, vs.z_q_x, vs.z_i_x, vs.x_tilde]
    r = sess.run(outs, feed_dict=feed)
    x_rec = r[-1]
    for i in range(10):
        o_x_i = x[i]
        r_x_i = x_rec[i]
        f, axarr = plt.subplots(1, 3)
        axarr[0].imshow(o_x_i[:, :, 0], cmap="gray")
        axarr[0].set_title("GT")
        axarr[1].imshow(r_x_i[:, :, 0], cmap="gray")
        axarr[1].set_title("REC")
        axarr[2].imshow(np.abs(r_x_i[:, :, 0] - o_x_i[:, :, 0]), cmap="gray")
        axarr[2].set_title("ABS DIFF REC")
        plt.savefig("vq_{}".format(i))
        plt.close()
