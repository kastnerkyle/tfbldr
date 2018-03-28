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
from scipy.io import wavfile


parser = argparse.ArgumentParser()
parser.add_argument('direct_model', nargs=1, default=None)
parser.add_argument('--seed', dest='seed', type=int, default=1999)
args = parser.parse_args()
direct_model = args.direct_model[0]

random_state = np.random.RandomState(args.seed)

config = tf.ConfigProto(
    device_count={'GPU': 0}
)

batch_size = 20
seq_length = 50
n_hid = 512
n_emb = 512
valid_random_state = np.random.RandomState(7)
# just to get char lookup
valid_itr = char_textfile_iterator("ptb_data/ptb.valid.txt", batch_size, seq_length,
                                   random_state=valid_random_state)
n_inputs = len(valid_itr.char2ind)
n_rounds = 3


sample_random_state = np.random.RandomState(1165)

with tf.Session(config=config) as sess:
    saver = tf.train.import_meta_graph(direct_model + '.meta')
    saver.restore(sess, direct_model)
    fields = ['inputs_tm1',
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
              'pred_sm']
    vs = namedtuple('Params', fields)(
        *[tf.get_collection(name)[0] for name in fields]
    )
    x = np.zeros((1, batch_size, 1))
    x = x + valid_itr.char2ind[" "]
    init_h = np.zeros((batch_size, n_hid)).astype("float32")
    init_c = np.zeros((batch_size, n_hid)).astype("float32")
    init_q_h = np.zeros((batch_size, n_hid)).astype("float32")
    init_q_c = np.zeros((batch_size, n_hid)).astype("float32")

    this_res = [x]
    this_i_res = [-1 * init_q_h[:, 0][None].astype("float32")]
    for i in range(seq_length):
        feed = {vs.inputs_tm1: x,
                vs.init_hidden: init_h,
                vs.init_cell: init_c,
                vs.init_q_hidden: init_q_h,
                vs.init_q_cell: init_q_c}
        outs = [vs.pred_sm, vs.hiddens, vs.cells, vs.q_hiddens, vs.q_cells, vs.i_hiddens]
        r = sess.run(outs, feed_dict=feed)
        p = r[0]
        # sample?
        #x = np.array([sample_random_state.choice(list(range(p.shape[-1])), p=p[0, i]) for i in range(batch_size)]).astype("float32")[None, :, None]
        x = p.argmax(axis=-1)[:, :, None].astype("float32")
        hids = r[1]
        cs = r[2]
        q_hids = r[3]
        q_cs = r[4]
        i_hids = r[5]
        init_h = hids[0]
        init_c = cs[0]
        init_q_h = q_hids[0]
        init_q_c = q_cs[0]
        this_res.append(x)
        this_i_res.append(i_hids)
    res = np.array(this_res)[:, 0, :, 0]
    i_res = np.array(this_i_res)[:, 0]

result_str = []
for bi in range(res.shape[1]):
    this_result_str = []
    for si in range(res.shape[0]):
        c = valid_itr.ind2char[int(res[si, bi])]
        this_result_str.append(c)
    result_str.append("".join(this_result_str))
from IPython import embed; embed(); raise ValueError()
