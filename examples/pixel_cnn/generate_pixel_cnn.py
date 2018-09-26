import argparse
import tensorflow as tf
import numpy as np
from tfbldr.datasets import fetch_fashion_mnist
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

fashion_mnist = fetch_fashion_mnist()
image_data = fashion_mnist["images"]# / 255.
label_data = fashion_mnist["target"]
# get images from the held out valid set
image_data = image_data[-10000:]
image_labels = label_data[-10000:][:, None]

random_state = np.random.RandomState(2000)
def sample_gumbel(logits, temperature=1.):
    noise = random_state.uniform(1E-5, 1. - 1E-5, np.shape(logits))
    #return np.argmax(np.log(softmax(logits, temperature)) - np.log(-np.log(noise)))
    return np.argmax(logits / float(temperature) - np.log(-np.log(noise)), axis=-1)

with tf.Session(config=config) as sess:
    saver = tf.train.import_meta_graph(model_path + '.meta')
    saver.restore(sess, model_path)
    fields = ['images',
              'labels',
              'x_tilde']
    vs = namedtuple('Params', fields)(
        *[tf.get_collection(name)[0] for name in fields]
    )
    x = image_data[:9]
    y = image_labels[:9]

    img = np.zeros((9, 28, 28))
    for i in range(img.shape[1]):
        for j in range(img.shape[2]):
            print("Sampling v completion pixel {}, {}".format(i, j))
            feed = {vs.images: img[..., None],
                    vs.labels: y}
            outs = [vs.x_tilde]
            r = sess.run(outs, feed_dict=feed)
            x_rec = sample_gumbel(r[-1])

            for k in range(img.shape[0]):
                if i < img.shape[1] // 2:
                    img[k, i, j] = float(x[k, i, j, 0])
                else:
                    img[k, i, j] = float(x_rec[k, i, j])

    f, axarr = plt.subplots(3, 3)
    for ii in range(len(axarr.flat)):
        axarr.flat[ii].imshow(img[ii], cmap="gray")
    plt.savefig("rec_v_mnist.png")
    plt.close()

    img = np.zeros((9, 28, 28))
    for i in range(img.shape[1]):
        for j in range(img.shape[2]):
            print("Sampling h completion pixel {}, {}".format(i, j))
            feed = {vs.images: img[..., None],
                    vs.labels: y}
            outs = [vs.x_tilde]
            r = sess.run(outs, feed_dict=feed)
            x_rec = sample_gumbel(r[-1])
            for k in range(img.shape[0]):
                if j < img.shape[1] // 2:
                    img[k, i, j] = float(x[k, i, j, 0])
                else:
                    img[k, i, j] = float(x_rec[k, i, j])

    f, axarr = plt.subplots(3, 3)
    for ii in range(len(axarr.flat)):
        axarr.flat[ii].imshow(img[ii], cmap="gray")
    plt.savefig("rec_h_mnist.png")
    plt.close()

    img = np.zeros((9, 28, 28))
    for i in range(img.shape[1]):
        for j in range(img.shape[2]):
            print("Sampling full completion pixel {}, {}".format(i, j))
            feed = {vs.images: img[..., None],
                    vs.labels: y}
            outs = [vs.x_tilde]
            r = sess.run(outs, feed_dict=feed)
            x_rec = sample_gumbel(r[-1])
            for k in range(img.shape[0]):
                img[k, i, j] = float(x_rec[k, i, j])

    f, axarr = plt.subplots(3, 3)
    for ii in range(len(axarr.flat)):
        axarr.flat[ii].imshow(img[ii], cmap="gray")
    plt.savefig("samp_mnist.png")
    plt.close()
    from IPython import embed; embed(); raise ValueError()
