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
              'inputs_t',
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
              'pred_sm',
              'per_step_rec_loss',
              'rec_loss',
              ]
    refs = []
    for name in fields:
        refs.append(tf.get_collection(name)[0])
    vs = namedtuple('Params', fields)(*refs)

    '''
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
        x = np.array([sample_random_state.choice(list(range(p.shape[-1])), p=p[0, i]) for i in range(batch_size)]).astype("float32")[None, :, None]
        #x = p.argmax(axis=-1)[:, :, None].astype("float32")
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
    print(result_str)
    '''

    # evaluate test set perplexity
    with open("ptb_data/ptb.test.txt", "rb") as f:
        lines = f.readlines()

    # do it the simple way
    tot = []
    ii = 0
    all_char_probs = []
    while True:
        current_indices = (ii * batch_size + np.arange(0, batch_size)).astype("int32")
        if current_indices[0] >= len(lines):
            break
        print("Processing lines {}:{} of total {}".format(current_indices[0] + 1, current_indices[-1] + 1, len(lines)))
        # start over on every line
        # lines always start with " "
        # so that is basically the SOS symbol
        these_lines = [lines[i] for i in current_indices if i in range(len(lines))]
        maxlen = max([len(line) for line in these_lines])
        x = np.zeros((maxlen, batch_size, 1)) - 1
        for bi in range(len(these_lines)):
            for li in range(len(these_lines[bi])):
                x[li, bi] = valid_itr.char2ind[these_lines[bi][li]]
        mask = (x >= 0).astype("int32")
        x = mask * x
        print("Minibatch shape {}".format(x.shape))

        x_tm1 = x[:-1]
        x_t = x[1:]

        init_h = np.zeros((batch_size, n_hid)).astype("float32")
        init_c = np.zeros((batch_size, n_hid)).astype("float32")
        init_q_h = np.zeros((batch_size, n_hid)).astype("float32")
        init_q_c = np.zeros((batch_size, n_hid)).astype("float32")

        feed = {vs.inputs_tm1: x_tm1,
                vs.inputs_t: x_t,
                vs.init_hidden: init_h,
                vs.init_cell: init_c,
                vs.init_q_hidden: init_q_h,
                vs.init_q_cell: init_q_c}
        outs = [vs.per_step_rec_loss, vs.pred_sm, vs.i_hiddens]
        r = sess.run(outs, feed_dict=feed)
        l = r[0]
        sm = r[1]
        masked_l = mask[1:, :, 0] * l
        char_probs = []
        for bi in range(len(these_lines)):
            x_t_ind = x_t[np.arange(len(these_lines[bi]) - 1), bi].flatten().astype("int32")
            cpr = sm[np.arange(len(these_lines[bi]) - 1), [bi] * (len(these_lines[bi]) - 1), x_t_ind]
            char_probs.append(cpr)
        all_char_probs += char_probs
        i_hids = r[2]
        ii += 1
    logsum_and_len = [(np.sum(np.log(acp)), len(acp)) for acp in all_char_probs]
    sumlogsum = np.sum([a[0] for a in logsum_and_len])
    sumlen = np.sum([a[1] for a in logsum_and_len])
    ppl = np.exp(-sumlogsum / sumlen)
    from IPython import embed; embed(); raise ValueError()
