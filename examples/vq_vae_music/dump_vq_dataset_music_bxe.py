import argparse
import tensorflow as tf
import numpy as np
from tfbldr.datasets import fetch_mnist
from collections import namedtuple
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy


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
idx = np.concatenate((np.arange(len(image_data))[:, None], np.arange(len(image_data))[:, None], np.arange(len(image_data))[:, None]), axis=1)
image_data = np.concatenate((image_data[..., 0][..., None],
                             image_data[..., 1][..., None],
                             image_data[..., 2][..., None]), axis=0)
idx = np.concatenate((idx[:, 0], idx[:, 1], idx[:, 2]), axis=0)
which_voice = np.zeros_like(idx)
which_voice[::3] = 0
which_voice[1::3] = 1
which_voice[2::3] = 2
shuffle_random = np.random.RandomState(112)
ii = np.arange(len(image_data))
shuffle_random.shuffle(ii)
image_data = image_data[ii]
idx = idx[ii]
which_voice = which_voice[ii] 

bs = 50
train_image_data = image_data[:-5000]
et = len(train_image_data) - len(train_image_data) % bs
train_image_data = train_image_data[:et]
train_idx = idx[:-5000]
train_idx = train_idx[:et]
train_which_voice = which_voice[:-5000]
train_which_voice = train_which_voice[:et]

valid_image_data = image_data[-5000:]
ev = len(valid_image_data) - len(valid_image_data) % bs
valid_image_data = valid_image_data[:ev]
valid_idx = idx[-5000:]
valid_idx = valid_idx[:ev]
valid_which_voice = which_voice[-5000:]
valid_which_voice = valid_which_voice[:ev]

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

train_conditions = []
twv = np.array(copy.deepcopy(train_which_voice))
tri = np.array(copy.deepcopy(train_idx))
ce = np.array(copy.deepcopy(d['centers']))

left_lu = tri - 1
left_lu[left_lu < 0] = 0
li = ce[left_lu]
left = np.array([lli[ttwv] for ttwv, lli in zip(twv, li)])

right_lu = tri + 1
right_lu[right_lu > max(tri)] = max(tri)
ri = ce[right_lu]
right = np.array([rri[ttwv] for ttwv, rri in zip(twv, ri)])

mid_lu = tri
mi = ce[mid_lu]
mid = np.array([mmi[ttwv] for ttwv, mmi in zip(twv, mi)])

train_conditions = list(zip(mid - left, mid - right))

vwv = np.array(copy.deepcopy(valid_which_voice))
vri = np.array(copy.deepcopy(valid_idx))

left_lu = vri - 1
left_lu[left_lu < 0] = 0
li = ce[left_lu]
left = np.array([lli[vvwv] for vvwv, lli in zip(vwv, li)])

right_lu = vri + 1
right_lu[right_lu > max(vri)] = max(vri)
ri = ce[right_lu]
right = np.array([rri[vvwv] for vvwv, rri in zip(vwv, ri)])

mid_lu = vri
mi = ce[mid_lu]
mid = np.array([mmi[vvwv] for vvwv, mmi in zip(vwv, mi)])

valid_conditions = list(zip(mid - left, mid - right))

mapper_values = sorted(list(set(train_conditions)) + [(None, None)])
condition_lookup = {v: k for k, v in enumerate(mapper_values)}
def mapper(c):
    return np.array([condition_lookup[ci] if ci in condition_lookup else condition_lookup[(None, None)] for ci in c])[:, None]

train_image_data = train_z_i
val_image_data = valid_z_i
train_labels = mapper(train_conditions)
valid_labels = mapper(valid_conditions)

np.savez("vq_vae_encoded_music.npz", train_z_i=train_z_i, valid_z_i=valid_z_i, train_conditions=train_conditions, valid_conditions=valid_conditions, train_labels=train_labels, valid_labels=valid_labels, mapper_values=mapper_values)
