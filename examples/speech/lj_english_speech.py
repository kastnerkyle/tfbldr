from __future__ import print_function
import os
import argparse
import numpy as np
import tensorflow as tf
from collections import namedtuple

import logging
import shutil
from tfbldr.datasets import rsync_fetch, fetch_ljspeech
from tfbldr.datasets import tbptt_file_list_iterator
from tfbldr.utils import next_experiment_path
from tfbldr import get_logger
from tfbldr import run_loop
from tfbldr.nodes import Linear
from tfbldr.nodes import LSTMCell
from tfbldr.nodes import GaussianAttentionCell
from tfbldr import scan

# TODO: add help info
parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', dest='seq_len', default=256, type=int)
parser.add_argument('--batch_size', dest='batch_size', default=64, type=int)
parser.add_argument('--epochs', dest='epochs', default=30, type=int)
parser.add_argument('--window_mixtures', dest='window_mixtures', default=10, type=int)
parser.add_argument('--lstm_layers', dest='lstm_layers', default=3, type=int)
parser.add_argument('--cell_dropout', dest='cell_dropout', default=.9, type=float)
parser.add_argument('--units_per_layer', dest='units', default=400, type=int)
parser.add_argument('--restore', dest='restore', default=None, type=str)
args = parser.parse_args()

ljspeech = rsync_fetch(fetch_ljspeech, "leto01")
files = ljspeech["file_paths"]
hybrid_lookup = ljspeech["vocabulary"]
hybrid_inverse_lookup = {v: k for k, v in hybrid_lookup.items()}
itr_random_state = np.random.RandomState(2177)
batch_size = args.batch_size
truncation_length = args.seq_len
cell_dropout_scale = args.cell_dropout
vocabulary_size = len(ljspeech["vocabulary"])
speech_size = 63

def file_access(f):
    d = np.load(f)
    text = d["text"]
    inds = [hybrid_lookup[t] for t in text.ravel()[0]]
    audio = d["audio_features"]
    return (audio, inds)

itr = tbptt_file_list_iterator(files, file_access,
                               batch_size,
                               truncation_length,
                               other_one_hot_size=[vocabulary_size],
                               random_state=itr_random_state)


# build normalization_stats off first 1000 minibatches
print("Building normalization stats...")
norm_r = []
steps = 1000
for i in range(steps):
    # normalization
    print("norm step {} of {}".format(i, steps))
    r = itr.next_batch()
    norm_r.append(r[0])

norm_arrs = np.concatenate(norm_r, axis=0)
norm_arrs = norm_arrs.reshape(-1, norm_arrs.shape[-1])
norm_arrs_mean = np.mean(norm_arrs, axis=0)
norm_arrs_std = np.std(norm_arrs, axis=0)
np.save("hybrid_mean", norm_arrs_mean)
np.save("hybrid_std", norm_arrs_std)
del itr

def file_access(f):
    d = np.load(f)
    text = d["text"]
    inds = [hybrid_lookup[t] for t in text.ravel()[0]]
    #audio = (d["audio_features"] - norm_arrs_mean[None, None]) / (norm_arrs_std[None, None])
    audio = d["audio_features"]
    return (audio, inds)

itr_random_state = np.random.RandomState(2177)
itr = tbptt_file_list_iterator(files, file_access,
                               batch_size,
                               truncation_length,
                               other_one_hot_size=[vocabulary_size],
                               random_state=itr_random_state)
epsilon = 1E-8
h_dim = args.units
forward_init = "truncated_normal"
rnn_init = "truncated_normal"
random_state = np.random.RandomState(1442)
window_mixtures = args.window_mixtures


def create_graph(num_letters,
                 speech_size,
                 batch_size,
                 num_units=400, lstm_layers=3,
                 window_mixtures=10):
    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(2899)

        speech = tf.placeholder(tf.float32, shape=[None, batch_size, speech_size])
        speech_mask = tf.placeholder(tf.float32, shape=[None, batch_size])

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
            in_speech = speech[:-1, :, :]
            in_speech_mask = speech_mask[:-1]
            out_speech = speech[1:, :, :]
            out_speech_mask = speech_mask[1:]

            def step(inp_t, inp_mask_t,
                     att_w_tm1, att_k_tm1, att_h_tm1, att_c_tm1,
                     h1_tm1, c1_tm1, h2_tm1, c2_tm1):

                o = GaussianAttentionCell([inp_t], [speech_size],
                                          (att_h_tm1, att_c_tm1),
                                          att_k_tm1,
                                          sequence,
                                          num_letters,
                                          num_units,
                                          att_w_tm1,
                                          input_mask=inp_mask_t,
                                          conditioning_mask=sequence_mask,
                                          attention_scale=1./10.,
                                          name="att",
                                          random_state=random_state,
                                          cell_dropout=cell_dropout,
                                          init=rnn_init)
                att_w_t, att_k_t, att_phi_t, s = o
                att_h_t = s[0]
                att_c_t = s[1]

                output, s = LSTMCell([inp_t, att_w_t, att_h_t],
                                     [speech_size, num_letters, num_units],
                                     h1_tm1, c1_tm1, num_units,
                                     input_mask=inp_mask_t,
                                     random_state=random_state,
                                     cell_dropout=cell_dropout,
                                     name="rnn1", init=rnn_init)
                h1_t = s[0]
                c1_t = s[1]

                output, s = LSTMCell([inp_t, att_w_t, h1_t],
                                     [speech_size, num_letters, num_units],
                                     h2_tm1, c2_tm1, num_units,
                                     input_mask=inp_mask_t,
                                     random_state=random_state,
                                     cell_dropout=cell_dropout,
                                     name="rnn2", init=rnn_init)
                h2_t = s[0]
                c2_t = s[1]
                return output, att_w_t, att_k_t, att_phi_t, att_h_t, att_c_t, h1_t, c1_t, h2_t, c2_t

            r = scan(step,
                     [in_speech, in_speech_mask],
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

            mean_pred = Linear([output], [num_units],
                               speech_size,
                               random_state=random_state,
                               init=forward_init,
                               name="mean_proj")
            loss = tf.reduce_mean(tf.square(mean_pred - out_speech))

            # save params for easier model loading and prediction
            for param in [('speech', speech),
                          ('in_speech', in_speech),
                          ('out_speech', out_speech),
                          ('speech_mask', speech_mask),
                          ('in_speech_mask', in_speech_mask),
                          ('out_speech_mask', out_speech_mask),
                          ('sequence', sequence),
                          ('sequence_mask', sequence_mask),
                          ('bias', bias),
                          ('cell_dropout', cell_dropout),
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
                          ('c2', c2),
                          ('mean_pred', mean_pred)]:
                tf.add_to_collection(*param)

            with tf.name_scope('training'):
                learning_rate = 0.0001
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, use_locking=True)
                grad, var = zip(*optimizer.compute_gradients(loss))
                grad, _ = tf.clip_by_global_norm(grad, 3.)
                train_step = optimizer.apply_gradients(zip(grad, var))

            with tf.name_scope('summary'):
                # TODO: add more summaries
                summary = tf.summary.merge([
                    tf.summary.scalar('loss', loss)
                ])

            things_names = ["speech",
                            "speech_mask",
                            "in_speech",
                            "in_speech_mask",
                            "out_speech",
                            "out_speech_mask",
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
                            "mean_pred",
                            "loss",
                            "train_step",
                            "learning_rate",
                            "summary"]
            things_tf = [speech,
                         speech_mask,
                         in_speech,
                         in_speech_mask,
                         out_speech,
                         out_speech_mask,
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
                         mean_pred,
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

    g, vs = create_graph(vocabulary_size,
                         speech_size,
                         batch_size,
                         num_units=args.units,
                         lstm_layers=args.lstm_layers,
                         window_mixtures=args.window_mixtures)

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
    loop_step = 0
    def loop(sess, itr, extras, stateful_args):
        speech, seq, reset = itr.next_batch()

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
        noise_pwr = 4.
        noise = noise_pwr * random_state.randn(*speech[:-1].shape)

        feed = {vs.in_speech: speech[:-1] + noise,
                vs.in_speech_mask: 0. * speech[:-1, :, 0] + 1.,
                vs.out_speech: speech[1:],
                vs.out_speech_mask: 0. * speech[1:, :, 0] + 1.,
                vs.sequence: seq,
                vs.sequence_mask: 0. * seq[:, :, 0] + 1.,
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
                 n_steps=500000,
                 n_train_steps_per=1000,
                 train_stateful_args=stateful_args,
                 n_valid_steps_per=0,
                 valid_stateful_args=stateful_args)


if __name__ == '__main__':
    main()
