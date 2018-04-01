import matplotlib
matplotlib.use("Agg")
import os

import argparse
import tensorflow as tf
import numpy as np
from tfbldr.datasets import char_textfile_iterator

from collections import namedtuple, defaultdict
import sys
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('direct_model', nargs=1, default=None)
parser.add_argument('--seed', dest='seed', type=int, default=1999)
args = parser.parse_args()
direct_model = args.direct_model[0]

random_state = np.random.RandomState(args.seed)

config = tf.ConfigProto(
    device_count={'GPU': 0}
)


# from Rithesh
ct_random = np.random.RandomState(1)
def copytask_loader(batch_size, seq_width, min_len, max_len):
    while True:
        # All batches have the same sequence length, but varies across batch
        if min_len == max_len:
            seq_len = min_len
        else:
            seq_len = ct_random.randint(min_len, max_len)
        seq = ct_random.binomial(1, 0.5, (seq_len, batch_size, seq_width))
        #seq = Variable(torch.from_numpy(seq)).cuda()

        # The input includes an additional channel used for the delimiter
        inp = np.zeros((2 * seq_len + 2, batch_size, seq_width + 1))
        inp[:seq_len, :, :seq_width] = seq
        inp[seq_len, :, seq_width] = 1.0 # start output delimiter in our control channel

        outp = np.zeros((2 * seq_len + 2, batch_size, seq_width + 1))
        outp[seq_len + 1:-1, :, :seq_width] = seq
        outp[-1, :, seq_width] = 1.0 # end output delimiter in our control channel
        yield inp.astype("float32"), outp.astype("float32")

batch_size = 10
seq_length_min = 5
seq_length_max = 20
seq_width = 8

n_hid = 512
n_emb = 512

copy_itr = copytask_loader(batch_size, seq_width, seq_length_min, seq_length_max)
a, b = next(copy_itr)
n_inputs = a.shape[-1]

with tf.Session(config=config) as sess:
    saver = tf.train.import_meta_graph(direct_model + '.meta')
    saver.restore(sess, direct_model)
    fields = ['inputs',
              'targets',
              'init_hidden',
              'init_cell',
              'init_q_hidden',
              'init_q_cell',
              'hiddens',
              'cells',
              'q_hiddens',
              'q_cells',
              'i_hiddens',
              'pred',
              'pred_sig',
              'rec_loss',
              ]
    refs = []
    for name in fields:
        refs.append(tf.get_collection(name)[0])
    vs = namedtuple('Params', fields)(*refs)

    init_h = np.zeros((batch_size, n_hid)).astype("float32")
    init_c = np.zeros((batch_size, n_hid)).astype("float32")
    init_q_h = np.zeros((batch_size, n_hid)).astype("float32")
    init_q_c = np.zeros((batch_size, n_hid)).astype("float32")

    inps, targets = next(copy_itr)

    feed = {vs.inputs: inps,
            vs.init_hidden: init_h,
            vs.init_cell: init_c,
            vs.init_q_hidden: init_q_h,
            vs.init_q_cell: init_q_c}
    outs = [vs.pred_sig, vs.hiddens, vs.cells, vs.q_hiddens, vs.q_cells, vs.i_hiddens]
    r = sess.run(outs, feed_dict=feed)

    n_to_show = 4
    f, axarr = plt.subplots(n_to_show, 5)

    for i in range(n_to_show):
        axarr[i, 0].matshow(inps[:, i, :])
        axarr[i, 1].matshow(targets[:, i, :])
        axarr[i, 2].matshow(r[0][:, i, :])
        axarr[i, 3].matshow(np.abs((r[0][:, i, :] > 0.5).astype("float32") - targets[:, i, :]))
        axarr[i, 4].stem(r[-1][:, i])
        axarr[i, 0].axis("off")
        axarr[i, 1].axis("off")
        axarr[i, 2].axis("off")
        axarr[i, 3].axis("off")
        axarr[i, 4].axis("off")
    plt.savefig("copy_res")
    from IPython import embed; embed(); raise ValueError()
