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

mnist = fetch_mnist()
image_data = mnist["images"] / 255.
# get images from the held out valid set
train_image_data = image_data[:-10000]
valid_image_data = image_data[-10000:]

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
    bs = 50
    assert len(train_image_data) % bs == 0
    assert len(valid_image_data) % bs == 0
    train_z_i = []
    for i in range(len(train_image_data) // bs):
        print("Train minibatch {}".format(i))
        x = train_image_data[i * bs:(i + 1) * bs]
        feed = {vs.images: x,
                vs.bn_flag: 1.}

        outs = [vs.z_e_x, vs.z_q_x, vs.z_i_x, vs.x_tilde]
        r = sess.run(outs, feed_dict=feed)
        x_rec = r[-1]
        z_i = r[-2]
        train_z_i += [zz[:, :, None] for zz in z_i]
    train_z_i = np.array(train_z_i)

    valid_z_i = []
    for i in range(len(valid_image_data) // bs):
        print("Valid minibatch {}".format(i))
        x = valid_image_data[i * bs:(i + 1) * bs]
        feed = {vs.images: x,
                vs.bn_flag: 1.}

        outs = [vs.z_e_x, vs.z_q_x, vs.z_i_x, vs.x_tilde]
        r = sess.run(outs, feed_dict=feed)
        x_rec = r[-1]
        z_i = r[-2]
        valid_z_i += [zz[:, :, None] for zz in z_i]
    valid_z_i = np.array(valid_z_i)

    np.savez("vq_vae_encoded_mnist.npz", train_z_i=train_z_i, valid_z_i=valid_z_i)
