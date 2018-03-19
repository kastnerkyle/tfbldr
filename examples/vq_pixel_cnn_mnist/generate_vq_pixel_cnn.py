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
parser.add_argument('pixelcnn_model', nargs=1, default=None)
parser.add_argument('vqvae_model', nargs=1, default=None)
parser.add_argument('--seed', dest='seed', type=int, default=1999)
args = parser.parse_args()
vqvae_model_path = args.vqvae_model[0]
pixelcnn_model_path = args.pixelcnn_model[0]

random_state = np.random.RandomState(args.seed)

mnist = fetch_mnist()
image_data = 0. * mnist["images"]# / 255.
label_data = mnist["target"]
# get images from the held out valid set
valid_image_data = image_data[-10000:]
valid_image_labels = label_data[-10000:][:, None]

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
    y = valid_image_labels[:64]

    pix_z = np.zeros((64, 7, 7))
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
    x = valid_image_data[:64]
    z_i = pix_z[:64]
    feed = {vs.images: 0. * x,
            vs.z_i_x: z_i,
            vs.bn_flag: 1.}
    outs = [vs.x_tilde]
    r = sess2.run(outs, feed_dict=feed)
    x_rec = r[-1]

f, axarr = plt.subplots(3, 3)
for ii in range(len(axarr.flat)):
    axarr.flat[ii].imshow(x_rec[ii][..., 0], cmap="gray")
plt.savefig("samp_vq_pixel_cnn_mnist.png")
plt.close()
