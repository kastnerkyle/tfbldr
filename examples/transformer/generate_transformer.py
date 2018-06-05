import argparse
import tensorflow as tf
import numpy as np
from tfbldr.datasets import fetch_mnist
from collections import namedtuple
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tfbldr.datasets import fetch_norvig_words
from tfbldr.datasets import list_iterator


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

norvig = fetch_norvig_words()
words = norvig["data"]
maxlen = max([len(words_i) for words_i in words])
word_length_limit = 10
words = [words_i for words_i in words if len(words_i) <= word_length_limit]
#vocab = "_0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
vocab = "_0123456789abcdefghijklmnopqrstuvwxyz"
v2i = {v: k for k, v in enumerate(vocab)}
i2v = {v: k for k, v in v2i.items()}
word_inds = [np.array([v2i[wi] for wi in word_i] + [0] * (word_length_limit - len(word_i)))[..., None] for word_i in words]
rev_word_inds = [np.array([v2i[wi] for wi in word_i][::-1] + [0] * (word_length_limit - len(word_i)))[..., None] for word_i in words]

train_itr_random_state = np.random.RandomState(1122)
valid_itr_random_state = np.random.RandomState(12)

random_state = np.random.RandomState(1999)

batch_size = 64
n_syms = len(vocab)

shuffled_inds = list(range(len(word_inds)))
train_itr_random_state.shuffle(shuffled_inds)
split = 250000
train_inds = shuffled_inds[:split]
valid_inds = shuffled_inds[split:]
train_word_inds = [word_inds[i] for i in train_inds]
train_rev_word_inds = [rev_word_inds[i] for i in train_inds]

valid_word_inds = [word_inds[i] for i in valid_inds]
valid_rev_word_inds = [rev_word_inds[i] for i in valid_inds]

train_itr = list_iterator([train_word_inds, train_rev_word_inds], batch_size, random_state=train_itr_random_state)
valid_itr = list_iterator([valid_word_inds, valid_rev_word_inds], batch_size, random_state=valid_itr_random_state)


with tf.Session(config=config) as sess:
    saver = tf.train.import_meta_graph(model_path + '.meta')
    saver.restore(sess, model_path)
    fields = ["inputs",
              "outputs",
              "outputs_masks",
              "pred_logits",
              "enc_atts_0",
              "enc_atts_1",
              "enc_atts_2",
              "dec_atts_0",
              "dec_atts_1",
              "dec_atts_2"]
    vs = namedtuple('Params', fields)(
        *[tf.get_collection(name)[0] for name in fields]
    )
    x, y = valid_itr.next_batch()
    x = x.transpose(1, 0, 2)
    y = y.transpose(1, 0, 2)

    new_y = np.zeros((y.shape[0] + 1, y.shape[1], y.shape[2]))
    new_y[1:] = y
    y_mask = np.zeros((y.shape[0] + 1, y.shape[1], y.shape[2]))
    y_mask[new_y > 0] = 1.
    y_mask[0] = 1.
    y_mask = y_mask[..., 0]
    y = new_y

    outputs = 0. * y
    outputs_masks = 0. * y_mask + 1
    #outputs = y
    #outputs_masks = y_mask
    for i in range(1, len(y)):
        feed = {vs.inputs: x,
                vs.outputs: outputs,
                vs.outputs_masks: outputs_masks}
        outs = [vs.pred_logits,
                vs.enc_atts_0,
                vs.enc_atts_1,
                vs.enc_atts_2,
                vs.dec_atts_0,
                vs.dec_atts_1,
                vs.dec_atts_2]
        r = sess.run(outs, feed_dict=feed)
        res = r[0]
        amax_pred = np.argmax(res, axis=-1)
        outputs[i, :, 0] = amax_pred[i - 1].ravel() #y[i].ravel()

    for mbi in range(outputs.shape[1]):
        y_i = y[:, mbi].ravel().astype("int32")
        x_i = x[:, mbi].ravel().astype("int32")
        o_i = outputs[:, mbi].ravel().astype("int32")
        x_s = "".join([i2v[ii] for ii in x_i if ii != 0])
        y_s = "".join([i2v[ii] for ii in y_i if ii != 0])
        o_s = "".join([i2v[ii] for ii in o_i])
        o_cut = [n for n, ss in enumerate(o_s) if ss == "_"]
        if len(o_cut) == 0 or len(o_cut) == 1:
            o_cut = None
        else:
            o_cut = o_cut[1]
        # 0th is always "_"
        o_s = o_s[1:o_cut]
        errs = sum([o_s[i] != y_s[i] for i in range(len(o_s))])
        print("Element {}: x {} | y {} | out {} | err {}".format(mbi, x_s, y_s, o_s, errs))
