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

batch_size = 128
seq_length = 100
n_hid = 1000
n_emb = 512
valid_random_state = np.random.RandomState(7)
# just to get char lookup
valid_itr = char_textfile_iterator("ptb_data/ptb.valid.txt", batch_size, seq_length,
                                   random_state=valid_random_state)
n_inputs = len(valid_itr.char2ind)
n_rounds = 3


sample_random_state = np.random.RandomState(1165)

with tf.Session() as sess:
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
    lines = sorted(lines, key=len, reverse=True)

    # do it the simple slow way
    tot = []
    ii = 0
    all_char_probs = []
    while True:
        if ii == len(lines):
            break
        print("Processing line {} of {}".format(ii + 1, len(lines)))
        line = lines[ii]
        x = np.zeros((len(line), batch_size, 1))
        for li in range(len(line)):
            x[li, 0] = valid_itr.char2ind[line[li]]

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
        outs = [vs.per_step_rec_loss, vs.pred_sm,
                vs.hiddens, vs.cells, vs.q_hiddens, vs.q_cells, vs.i_hiddens]
        r = sess.run(outs, feed_dict=feed)
        l = r[0]
        sm = r[1]
        hiddens = r[2]
        cells = r[3]
        q_hiddens = r[4]
        q_cells = r[5]
        init_h = hiddens[-1]
        init_c = cells[-1]
        init_q_h = q_hiddens[-1]
        init_q_c = q_cells[-1]
        x_t_ind = x_t[:, 0, 0].astype("int32")
        cpr = sm[np.arange(len(x) - 1), 0, x_t_ind]
        all_char_probs += [cpr]
        ii += 1
    # http://ofir.io/Neural-Language-Modeling-From-Scratch/
    # this isn't right...
    logsum_and_len = [(np.sum(np.log(acp)), len(acp)) for acp in all_char_probs]
    sumlogsum = np.sum([a[0] for a in logsum_and_len])
    sumlen = np.sum([a[1] for a in logsum_and_len])
    ppl = np.power(2, -sumlogsum / sumlen)
    # https://stats.stackexchange.com/questions/211858/how-to-compute-bits-per-character-bpc
    log2sum_and_len = [(np.sum(np.log2(acp)), len(acp)) for acp in all_char_probs]
    sumlog2sum = np.sum([a[0] for a in log2sum_and_len])
    sumlen = np.sum([a[1] for a in log2sum_and_len])
    bpc = -sumlog2sum / sumlen
    print("BPC: {}".format(bpc))
    from IPython import embed; embed(); raise ValueError()
