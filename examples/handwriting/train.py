from __future__ import print_function
import os
import argparse
import numpy as np
import tensorflow as tf
from collections import namedtuple

import logging
import shutil
from tfbldr.datasets import rsync_fetch, fetch_iamondb
from tfbldr.datasets import tbptt_list_iterator
from tfbldr.utils import next_experiment_path
from tfbldr import get_logger
from tfbldr import run_loop
from tfbldr.nodes import Linear
from tfbldr.nodes import Linear
from tfbldr.nodes import LSTMCell
from tfbldr.nodes import GaussianAttentionCell
from tfbldr.nodes import BernoulliAndCorrelatedGMM
from tfbldr.nodes import BernoulliAndCorrelatedGMMCost
from tfbldr import scan

# TODO: add help info
parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', dest='seq_len', default=256, type=int)
parser.add_argument('--batch_size', dest='batch_size', default=64, type=int)
parser.add_argument('--epochs', dest='epochs', default=30, type=int)
parser.add_argument('--window_mixtures', dest='window_mixtures', default=10, type=int)
parser.add_argument('--output_mixtures', dest='output_mixtures', default=20, type=int)
parser.add_argument('--lstm_layers', dest='lstm_layers', default=3, type=int)
parser.add_argument('--cell_dropout', dest='cell_dropout', default=.9, type=float)
parser.add_argument('--units_per_layer', dest='units', default=400, type=int)
parser.add_argument('--restore', dest='restore', default=None, type=str)
args = parser.parse_args()

iamondb = rsync_fetch(fetch_iamondb, "leto01")
trace_data = iamondb["data"]
char_data = iamondb["target"]
batch_size = args.batch_size
truncation_len = args.seq_len
cell_dropout_scale = args.cell_dropout
vocabulary_size = len(iamondb["vocabulary"])
itr_random_state = np.random.RandomState(2177)
itr = tbptt_list_iterator(trace_data, [char_data], batch_size, truncation_len,
                          other_one_hot_size=[vocabulary_size],
                          random_state=itr_random_state)
epsilon = 1E-8

h_dim = args.units
forward_init = "truncated_normal"
rnn_init = "truncated_normal"
random_state = np.random.RandomState(1442)
output_mixtures = args.output_mixtures
window_mixtures = args.window_mixtures


def create_graph(num_letters, batch_size,
                 num_units=400, lstm_layers=3,
                 window_mixtures=10, output_mixtures=20):
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(2899)

        coordinates = tf.placeholder(tf.float32, shape=[None, batch_size, 3])
        coordinates_mask = tf.placeholder(tf.float32, shape=[None, batch_size])

        sequence = tf.placeholder(tf.float32, shape=[None, batch_size, num_letters])
        sequence_mask = tf.placeholder(tf.float32, shape=[None, batch_size])

        bias = tf.placeholder_with_default(tf.zeros(shape=[]), shape=[])
        cell_dropout = tf.placeholder_with_default(cell_dropout_scale * tf.ones(shape=[]), shape=[])
        att_w_init = tf.placeholder(tf.float32, shape=[batch_size, num_letters])
        att_k_init = tf.placeholder(tf.float32, shape=[batch_size, window_mixtures])
        att_h_init = tf.placeholder(tf.float32, shape=[batch_size, num_units])
        att_c_init = tf.placeholder(tf.float32, shape=[batch_size, num_units])
        h1_init = tf.placeholder(tf.float32, shape=[batch_size, num_units])
        c1_init = tf.placeholder(tf.float32, shape=[batch_size, num_units])
        h2_init = tf.placeholder(tf.float32, shape=[batch_size, num_units])
        c2_init = tf.placeholder(tf.float32, shape=[batch_size, num_units])

        def create_model():
            in_coordinates = coordinates[:-1, :, :]
            in_coordinates_mask = coordinates_mask[:-1]
            out_coordinates = coordinates[1:, :, :]
            out_coordinates_mask = coordinates_mask[1:]

            def step(inp_t, inp_mask_t,
                     att_w_tm1, att_k_tm1, att_h_tm1, att_c_tm1,
                     h1_tm1, c1_tm1, h2_tm1, c2_tm1):

                o = GaussianAttentionCell([inp_t], [3],
                                          (att_h_tm1, att_c_tm1),
                                          att_k_tm1,
                                          sequence,
                                          num_letters,
                                          num_units,
                                          att_w_tm1,
                                          input_mask=inp_mask_t,
                                          conditioning_mask=sequence_mask,
                                          attention_scale = 1. / 25.,
                                          name="att",
                                          random_state=random_state,
                                          cell_dropout=cell_dropout,
                                          init=rnn_init)
                att_w_t, att_k_t, att_phi_t, s = o
                att_h_t = s[0]
                att_c_t = s[1]

                output, s = LSTMCell([inp_t, att_w_t, att_h_t],
                                     [3, num_letters, num_units],
                                     h1_tm1, c1_tm1, num_units,
                                     input_mask=inp_mask_t,
                                     random_state=random_state,
                                     cell_dropout=cell_dropout,
                                     name="rnn1", init=rnn_init)
                h1_t = s[0]
                c1_t = s[1]

                output, s = LSTMCell([inp_t, att_w_t, h1_t],
                                     [3, num_letters, num_units],
                                     h2_tm1, c2_tm1, num_units,
                                     input_mask=inp_mask_t,
                                     random_state=random_state,
                                     cell_dropout=cell_dropout,
                                     name="rnn2", init=rnn_init)
                h2_t = s[0]
                c2_t = s[1]
                return output, att_w_t, att_k_t, att_phi_t, att_h_t, att_c_t, h1_t, c1_t, h2_t, c2_t

            r = scan(step,
                     [in_coordinates, in_coordinates_mask],
                     [None, att_w_init, att_k_init, None, att_h_init, att_c_init,
                      h1_init, c1_init, h2_init, c2_init])
            output = r[0]
            att_w = r[1]
            att_k = r[2]
            att_phi = r[3]
            att_h = r[4]
            att_c = r[5]
            h1 = r[6]
            c1 = r[7]
            h2 = r[8]
            c2 = r[9]


            #output = tf.reshape(output, [-1, num_units])
            mo = BernoulliAndCorrelatedGMM([output], [num_units],
                                           bias=bias, n_components=output_mixtures,
                                           random_state=random_state,
                                           init=forward_init,
                                           name="split")
            e, pi, mu1, mu2, std1, std2, rho = mo

            #coords = tf.reshape(out_coordinates, [-1, 3])
            #xs, ys, es = tf.unstack(tf.expand_dims(coords, axis=2), axis=1)

            xs = out_coordinates[..., 0][..., None]
            ys = out_coordinates[..., 1][..., None]
            es = out_coordinates[..., 2][..., None]

            cc = BernoulliAndCorrelatedGMMCost(e, pi,
                                               [mu1, mu2],
                                               [std1, std2],
                                               rho,
                                               es,
                                               [xs, ys],
                                               name="cost")
            loss = tf.reduce_mean(cc)

            # save params for easier model loading and prediction
            for param in [('coordinates', coordinates),
                          ('in_coordinates', in_coordinates),
                          ('out_coordinates', out_coordinates),
                          ('coordinates_mask', coordinates_mask),
                          ('in_coordinates_mask', in_coordinates_mask),
                          ('out_coordinates_mask', out_coordinates_mask),
                          ('sequence', sequence),
                          ('sequence_mask', sequence_mask),
                          ('bias', bias),
                          ('cell_dropout', cell_dropout),
                          ('e', e), ('pi', pi),
                          ('mu1', mu1), ('mu2', mu2),
                          ('std1', std1), ('std2', std2),
                          ('rho', rho),
                          ('att_w_init', att_w_init),
                          ('att_k_init', att_k_init),
                          ('att_h_init', att_h_init),
                          ('att_c_init', att_c_init),
                          ('h1_init', h1_init),
                          ('c1_init', c1_init),
                          ('h2_init', h2_init),
                          ('c2_init', c2_init),
                          ('att_w', att_w),
                          ('att_k', att_k),
                          ('att_phi', att_phi),
                          ('att_h', att_h),
                          ('att_c', att_c),
                          ('h1', h1),
                          ('c1', c1),
                          ('h2', h2),
                          ('c2', c2)]:
                tf.add_to_collection(*param)

            with tf.name_scope('training'):
                steps = tf.Variable(0.)
                learning_rate = tf.train.exponential_decay(0.001, steps, staircase=True,
                                                           decay_steps=10000, decay_rate=0.5)

                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, use_locking=True)
                grad, var = zip(*optimizer.compute_gradients(loss))
                grad, _ = tf.clip_by_global_norm(grad, 3.)
                train_step = optimizer.apply_gradients(zip(grad, var), global_step=steps)

            with tf.name_scope('summary'):
                # TODO: add more summaries
                summary = tf.summary.merge([
                    tf.summary.scalar('loss', loss)
                ])

            things_names = ["coordinates",
                            "coordinates_mask",
                            "sequence",
                            "sequence_mask",
                            "att_w_init",
                            "att_k_init",
                            "att_h_init",
                            "att_c_init",
                            "h1_init",
                            "c1_init",
                            "h2_init",
                            "c2_init",
                            "att_w",
                            "att_k",
                            "att_phi",
                            "att_h",
                            "att_c",
                            "h1",
                            "c1",
                            "h2",
                            "c2",
                            "loss",
                            "train_step",
                            "learning_rate",
                            "summary"]
            things_tf = [coordinates,
                         coordinates_mask,
                         sequence,
                         sequence_mask,
                         att_w_init,
                         att_k_init,
                         att_h_init,
                         att_c_init,
                         h1_init,
                         c1_init,
                         h2_init,
                         c2_init,
                         att_w,
                         att_k,
                         att_phi,
                         att_h,
                         att_c,
                         h1,
                         c1,
                         h2,
                         c2,
                         loss,
                         train_step,
                         learning_rate,
                         summary]
            return namedtuple('Model', things_names)(*things_tf)

        train_model = create_model()
    return graph, train_model


def main():
    restore_model = args.restore
    seq_len = args.seq_len
    batch_size = args.batch_size
    num_epoch = args.epochs
    num_units = args.units
    batches_per_epoch = 1000

    g, vs = create_graph(vocabulary_size, batch_size,
                         num_units=args.units, lstm_layers=args.lstm_layers,
                         window_mixtures=args.window_mixtures,
                         output_mixtures=args.output_mixtures)

    num_letters = vocabulary_size
    att_w_init = np.zeros((batch_size, num_letters))
    att_k_init = np.zeros((batch_size, window_mixtures))
    att_h_init = np.zeros((batch_size, num_units))
    att_c_init = np.zeros((batch_size, num_units))
    h1_init = np.zeros((batch_size, num_units))
    c1_init = np.zeros((batch_size, num_units))
    h2_init = np.zeros((batch_size, num_units))
    c2_init = np.zeros((batch_size, num_units))

    stateful_args = [att_w_init,
                     att_k_init,
                     att_h_init,
                     att_c_init,
                     h1_init,
                     c1_init,
                     h2_init,
                     c2_init]

    def loop(sess, itr, extras, stateful_args):
        coords, coords_mask, seq, seq_mask, reset = itr.next_masked_batch()

        att_w_init = stateful_args[0]
        att_k_init = stateful_args[1]
        att_h_init = stateful_args[2]
        att_c_init = stateful_args[3]
        h1_init = stateful_args[4]
        c1_init = stateful_args[5]
        h2_init = stateful_args[6]
        c2_init = stateful_args[7]

        att_w_init *= reset
        att_k_init *= reset
        att_h_init *= reset
        att_c_init *= reset
        h1_init *= reset
        c1_init *= reset
        h2_init *= reset
        c2_init *= reset

        feed = {vs.coordinates: coords,
                vs.coordinates_mask: coords_mask,
                vs.sequence: seq,
                vs.sequence_mask: seq_mask,
                vs.att_w_init: att_w_init,
                vs.att_k_init: att_k_init,
                vs.att_h_init: att_h_init,
                vs.att_c_init: att_c_init,
                vs.h1_init: h1_init,
                vs.c1_init: c1_init,
                vs.h2_init: h2_init,
                vs.c2_init: c2_init}
        outs = [vs.att_w, vs.att_k,
                vs.att_h, vs.att_c,
                vs.h1, vs.c1, vs.h2, vs.c2,
                vs.att_phi,
                vs.loss, vs.summary, vs.train_step]
        r = sess.run(outs, feed_dict=feed)

        att_w_np = r[0]
        att_k_np = r[1]
        att_h_np = r[2]
        att_c_np = r[3]
        h1_np = r[4]
        c1_np = r[5]
        h2_np = r[6]
        c2_np = r[7]
        att_phi_np = r[8]
        l = r[-3]
        s = r[-2]
        _ = r[-1]

        # set next inits
        att_w_init = att_w_np[-1]
        att_k_init = att_k_np[-1]
        att_h_init = att_h_np[-1]
        att_c_init = att_c_np[-1]
        h1_init = h1_np[-1]
        c1_init = c1_np[-1]
        h2_init = h2_np[-1]
        c2_init = c2_np[-1]

        stateful_args = [att_w_init,
                         att_k_init,
                         att_h_init,
                         att_c_init,
                         h1_init,
                         c1_init,
                         h2_init,
                         c2_init]
        return l, s, stateful_args

    with tf.Session(graph=g) as sess:
        run_loop(sess,
                 loop, itr,
                 loop, itr,
                 n_train_steps_per=1000,
                 train_stateful_args=stateful_args,
                 n_valid_steps_per=0,
                 valid_stateful_args=stateful_args)


if __name__ == '__main__':
    main()
