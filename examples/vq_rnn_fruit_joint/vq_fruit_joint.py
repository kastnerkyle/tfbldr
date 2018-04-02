from tfbldr.nodes import Conv2d
from tfbldr.nodes import ConvTranspose2d
from tfbldr.nodes import VqEmbedding
from tfbldr.nodes import BatchNorm2d
from tfbldr.nodes import Linear
from tfbldr.nodes import ReLU
from tfbldr.nodes import Sigmoid
from tfbldr.nodes import Tanh
from tfbldr.nodes import OneHot
from tfbldr.nodes import Softmax
from tfbldr.nodes import LSTMCell
from tfbldr.nodes import CategoricalCrossEntropyIndexCost
from tfbldr.nodes import CategoricalCrossEntropyLinearIndexCost
from tfbldr.nodes import BernoulliCrossEntropyCost
from tfbldr.datasets import ordered_list_iterator
from tfbldr.plot import get_viridis
from tfbldr.plot import autoaspect
from tfbldr.datasets import fetch_fruitspeech
from tfbldr import get_params_dict
from tfbldr import run_loop
from tfbldr import scan
import tensorflow as tf
import numpy as np
from collections import namedtuple, defaultdict
import itertools

viridis_cm = get_viridis()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fruit = fetch_fruitspeech()
minmin = np.inf
maxmax = -np.inf

for s in fruit["data"]:
    si = s - s.mean()
    minmin = min(minmin, si.min())
    maxmax = max(maxmax, si.max())

train_data = []
valid_data = []
type_counts = defaultdict(lambda: 0)
final_audio = []

for n, s in enumerate(fruit["data"]):
    type_counts[fruit["target"][n]] += 1
    s = s - s.mean()
    n_s = (s - minmin) / float(maxmax - minmin)
    n_s = 2 * n_s - 1
    #n_s = mu_law_transform(n_s, 256)
    if type_counts[fruit["target"][n]] == 15:
        valid_data.append(n_s)
    else:
        train_data.append(n_s)


def _cuts(list_of_audio, cut, step):
    # make many overlapping cuts
    # 8k, this means offset is ~4ms @ step of 32
    real_final = []
    real_idx = []
    for n, s in enumerate(list_of_audio):
        # cut off the end
        s = s[:len(s) - len(s) % step]
        starts = np.arange(0, len(s) - cut + step, step)
        for st in starts:
            real_final.append(s[st:st + cut][None, :, None])
            real_idx.append(n)
    return real_final, real_idx

cut = 256
step = 1
train_audio, train_audio_idx = _cuts(train_data, cut, step)
valid_audio, valid_audio_idx = _cuts(valid_data, cut, step)

random_state = np.random.RandomState(1999)
l1_dim = (64, 1, 4, [1, 1, 2, 1])
l2_dim = (128, 1, 4, [1, 1, 2, 1])
l3_dim = (256, 1, 4, [1, 1, 2, 1])
l3_dim = (257, 1, 4, [1, 1, 2, 1])
l4_dim = (256, 1, 4, [1, 1, 2, 1])
l5_dim = (257, 1, 1, [1, 1, 1, 1])
embedding_dim = 512
vqvae_batch_size = 50
rnn_batch_size = 50
n_hid = 512
n_clusters = 64
# goes from 256 -> 16
hardcoded_z_len = 16
# reserve 0 for "start code"
n_inputs = embedding_dim + 1
switch_step = 10000
both = True
# reserve 0 for start code

rnn_init = "truncated_normal"
forward_init = "truncated_normal"

l_dims = [l1_dim, l2_dim, l3_dim, l4_dim, l5_dim]
stride_div = np.prod([ld[-1] for ld in l_dims])
ebpad = [0, 0, 4 // 2 - 1, 0]
dbpad = [0, 0, 4 // 2 - 1, 0]

train_itr_random_state = np.random.RandomState(1122)
valid_itr_random_state = np.random.RandomState(12)
train_itr = ordered_list_iterator([train_audio], train_audio_idx, vqvae_batch_size, random_state=train_itr_random_state)
valid_itr = ordered_list_iterator([valid_audio], valid_audio_idx, vqvae_batch_size, random_state=valid_itr_random_state)

"""
for i in range(10000):
    tt = train_itr.next_batch()
    # tt[0][3][:, :16] == tt[0][2][:, 16:32]
"""


def create_encoder(inp, bn_flag):
    l1 = Conv2d([inp], [1], l_dims[0][0], kernel_size=l_dims[0][1:3], name="enc1",
                strides=l_dims[0][-1],
                border_mode=ebpad,
                random_state=random_state)
    bn_l1 = BatchNorm2d(l1, bn_flag, name="bn_enc1")
    r_l1 = ReLU(bn_l1)

    l2 = Conv2d([r_l1], [l_dims[0][0]], l_dims[1][0], kernel_size=l_dims[1][1:3], name="enc2",
                strides=l_dims[1][-1],
                border_mode=ebpad,
                random_state=random_state)
    bn_l2 = BatchNorm2d(l2, bn_flag, name="bn_enc2")
    r_l2 = ReLU(bn_l2)

    l3 = Conv2d([r_l2], [l_dims[1][0]], l_dims[2][0], kernel_size=l_dims[2][1:3], name="enc3",
                strides=l_dims[2][-1],
                border_mode=ebpad,
                random_state=random_state)
    bn_l3 = BatchNorm2d(l3, bn_flag, name="bn_enc3")
    r_l3 = ReLU(bn_l3)

    l4 = Conv2d([r_l3], [l_dims[2][0]], l_dims[3][0], kernel_size=l_dims[3][1:3], name="enc4",
                strides=l_dims[3][-1],
                border_mode=ebpad,
                random_state=random_state)
    bn_l4 = BatchNorm2d(l4, bn_flag, name="bn_enc4")
    r_l4 = ReLU(bn_l4)

    l5 = Conv2d([r_l4], [l_dims[3][0]], l_dims[4][0], kernel_size=l_dims[4][1:3], name="enc5",
                random_state=random_state)
    bn_l5 = BatchNorm2d(l5, bn_flag, name="bn_enc5")
    return bn_l5


def create_decoder(latent, bn_flag):
    l1 = Conv2d([latent], [l_dims[-1][0]], l_dims[-2][0], kernel_size=l_dims[-1][1:3], name="dec1",
                random_state=random_state)
    bn_l1 = BatchNorm2d(l1, bn_flag, name="bn_dec1")
    r_l1 = ReLU(bn_l1)

    l2 = ConvTranspose2d([r_l1], [l_dims[-2][0]], l_dims[-3][0], kernel_size=l_dims[-2][1:3], name="dec2",
                         strides=l_dims[-2][-1],
                         border_mode=dbpad,
                         random_state=random_state)
    bn_l2 = BatchNorm2d(l2, bn_flag, name="bn_dec2")
    r_l2 = ReLU(bn_l2)

    l3 = ConvTranspose2d([r_l2], [l_dims[-3][0]], l_dims[-4][0], kernel_size=l_dims[-3][1:3], name="dec3",
                         strides=l_dims[-3][-1],
                         border_mode=dbpad,
                         random_state=random_state)
    bn_l3 = BatchNorm2d(l3, bn_flag, name="bn_dec3")
    r_l3 = ReLU(bn_l3)

    l4 = ConvTranspose2d([r_l3], [l_dims[-4][0]], l_dims[-5][0], kernel_size=l_dims[-4][1:3], name="dec4",
                         strides=l_dims[-4][-1],
                         border_mode=dbpad,
                         random_state=random_state)
    bn_l4 = BatchNorm2d(l4, bn_flag, name="bn_dec4")
    r_l4 = ReLU(bn_l4)


    l5 = ConvTranspose2d([r_l4], [l_dims[-5][0]], 1, kernel_size=l_dims[-5][1:3], name="dec5",
                         strides=l_dims[-5][-1],
                         border_mode=dbpad,
                         random_state=random_state)
    #s_l5 = Sigmoid(l5)
    t_l5 = Tanh(l5)
    return t_l5


def create_vqvae(inp, bn):
    z_e_x = create_encoder(inp, bn)
    z_q_x, z_i_x, z_nst_q_x, emb = VqEmbedding(z_e_x, l_dims[-1][0], embedding_dim, random_state=random_state, name="embed")
    x_tilde = create_decoder(z_q_x, bn)
    return x_tilde, z_e_x, z_q_x, z_i_x, z_nst_q_x, emb


def create_vqrnn(inp_tm1, inp_t, h1_init, c1_init, h1_q_init, c1_q_init):
    oh_tm1 = OneHot(inp_tm1, n_inputs)
    p_tm1 = Linear([oh_tm1], [n_inputs], n_hid, random_state=random_state, name="proj",
                  init=forward_init)
    def step(x_t, h1_tm1, c1_tm1, h1_q_tm1, c1_q_tm1):
        output, s = LSTMCell([x_t], [n_hid], h1_tm1, c1_tm1, n_hid,
                             random_state=random_state,
                             name="rnn1", init=rnn_init)
        h1_t = s[0]
        c1_t = s[1]

        output, s = LSTMCell([h1_t], [n_hid], h1_q_tm1, c1_q_tm1, n_hid,
                             random_state=random_state,
                             name="rnn1_q", init=rnn_init)
        h1_cq_t = s[0]
        c1_q_t = s[1]

        h1_q_t, h1_i_t, h1_nst_q_t, h1_emb = VqEmbedding(h1_cq_t, n_hid, n_clusters,
                                                         random_state=random_state,
                                                         name="h1_vq_emb")

        # not great
        h1_i_t = tf.cast(h1_i_t, tf.float32)
        return output, h1_t, c1_t, h1_q_t, c1_q_t, h1_nst_q_t, h1_cq_t, h1_i_t

    r = scan(step, [p_tm1], [None, h1_init, c1_init, h1_q_init, c1_q_init, None, None, None])
    out = r[0]
    hiddens = r[1]
    cells = r[2]
    q_hiddens = r[3]
    q_cells = r[4]
    q_nst_hiddens = r[5]
    q_nvq_hiddens  = r[6]
    i_hiddens = r[7]

    pred = Linear([out], [n_hid], n_inputs, random_state=random_state, name="out",
                  init=forward_init)
    pred_sm = Softmax(pred)
    return pred_sm, pred, hiddens, cells, q_hiddens, q_cells, q_nst_hiddens, q_nvq_hiddens, i_hiddens, oh_tm1


def create_graph():
    graph = tf.Graph()
    with graph.as_default():
        # vqvae part
        # define all the vqvae inputs and outputs
        vqvae_inputs = tf.placeholder(tf.float32, shape=[None, train_audio[0].shape[0],
                                                         train_audio[0].shape[1],
                                                         train_audio[0].shape[2]])
        bn_flag = tf.placeholder_with_default(tf.zeros(shape=[]), shape=[])
        x_tilde, z_e_x, z_q_x, z_i_x, z_nst_q_x, z_emb = create_vqvae(vqvae_inputs, bn_flag)

        #rec_loss = tf.reduce_mean(BernoulliCrossEntropyCost(x_tilde, images))
        vqvae_rec_loss = tf.reduce_mean(tf.square(x_tilde - vqvae_inputs))
        vqvae_vq_loss = tf.reduce_mean(tf.square(tf.stop_gradient(z_e_x) - z_nst_q_x))
        vqvae_commit_loss = tf.reduce_mean(tf.square(z_e_x - tf.stop_gradient(z_nst_q_x)))
        vqvae_alpha = 1.
        vqvae_beta = 0.25
        vqvae_loss = vqvae_rec_loss + vqvae_alpha * vqvae_vq_loss + vqvae_beta * vqvae_commit_loss
        vqvae_params = get_params_dict()
        # get vqvae keys now, dict is *dynamic* and shared
        vqvae_params_keys = [k for k in vqvae_params.keys()]
        vqvae_grads = tf.gradients(vqvae_loss, vqvae_params.values())

        learning_rate = 0.0002
        vqvae_optimizer = tf.train.AdamOptimizer(learning_rate, use_locking=True)
        assert len(vqvae_grads) == len(vqvae_params)
        j = [(g, p) for g, p in zip(vqvae_grads, vqvae_params.values())]
        vqvae_train_step = vqvae_optimizer.apply_gradients(j)

        # rnn part
        # ultimately we will use 2 calls to feed_dict to make lookup mappings easier, but could do it like this
        #rnn_inputs = tf.cast(tf.stop_gradient(tf.transpose(z_i_x, (2, 0, 1))), tf.float32)
        rnn_inputs = tf.placeholder(tf.float32, shape=[None, rnn_batch_size, 1])
        rnn_inputs_tm1 = rnn_inputs[:-1]
        rnn_inputs_t = rnn_inputs[1:]

        init_hidden = tf.placeholder(tf.float32, shape=[rnn_batch_size, n_hid])
        init_cell = tf.placeholder(tf.float32, shape=[rnn_batch_size, n_hid])
        init_q_hidden = tf.placeholder(tf.float32, shape=[rnn_batch_size, n_hid])
        init_q_cell = tf.placeholder(tf.float32, shape=[rnn_batch_size, n_hid])
        r = create_vqrnn(rnn_inputs_tm1, rnn_inputs_t, init_hidden, init_cell, init_q_hidden, init_q_cell)
        pred_sm, pred, hiddens, cells, q_hiddens, q_cells, q_nst_hiddens, q_nvq_hiddens, i_hiddens, oh_tm1 = r

        rnn_rec_loss = tf.reduce_mean(CategoricalCrossEntropyIndexCost(pred_sm, rnn_inputs_t))
        #rnn_rec_loss = tf.reduce_mean(CategoricalCrossEntropyLinearIndexCost(pred, rnn_inputs_t))

        rnn_alpha = 1.
        rnn_beta = 0.25
        rnn_vq_h_loss = tf.reduce_mean(tf.square(tf.stop_gradient(q_nvq_hiddens) - q_nst_hiddens))
        rnn_commit_h_loss = tf.reduce_mean(tf.square(q_nvq_hiddens - tf.stop_gradient(q_nst_hiddens)))

        rnn_loss = rnn_rec_loss + rnn_alpha * rnn_vq_h_loss + rnn_beta * rnn_commit_h_loss
        rnn_params = {k:v for k, v in get_params_dict().items() if k not in vqvae_params_keys}
        rnn_grads = tf.gradients(rnn_loss, rnn_params.values())
        learning_rate = 0.0001
        rnn_optimizer = tf.train.AdamOptimizer(learning_rate, use_locking=True)
        assert len(rnn_grads) == len(rnn_params)
        rnn_grads = [tf.clip_by_value(g, -10., 10.) if g is not None else None for g in rnn_grads]
        j = [(g, p) for g, p in zip(rnn_grads, rnn_params.values())]
        rnn_train_step = rnn_optimizer.apply_gradients(j)

    things_names = ["vqvae_inputs",
                    "bn_flag",
                    "x_tilde",
                    "z_e_x",
                    "z_q_x",
                    "z_i_x",
                    "z_emb",
                    "vqvae_loss",
                    "vqvae_rec_loss",
                    "vqvae_train_step",
                    "rnn_inputs",
                    "rnn_inputs_tm1",
                    "rnn_inputs_t",
                    "init_hidden",
                    "init_cell",
                    "init_q_hidden",
                    "init_q_cell",
                    "hiddens",
                    "cells",
                    "q_hiddens",
                    "q_cells",
                    "q_nvq_hiddens",
                    "i_hiddens",
                    "pred",
                    "pred_sm",
                    "oh_tm1",
                    "rnn_loss",
                    "rnn_rec_loss",
                    "rnn_train_step"]

    things_tf = [eval(name) for name in things_names]
    for tn, tt in zip(things_names, things_tf):
        graph.add_to_collection(tn, tt)
    train_model = namedtuple('Model', things_names)(*things_tf)
    return graph, train_model

g, vs = create_graph()

rnn_train = False
step = 0

def loop(sess, itr, extras, stateful_args):
    x, = itr.next_batch()

    init_h = np.zeros((rnn_batch_size, n_hid)).astype("float32")
    init_c = np.zeros((rnn_batch_size, n_hid)).astype("float32")
    init_q_h = np.zeros((rnn_batch_size, n_hid)).astype("float32")
    init_q_c = np.zeros((rnn_batch_size, n_hid)).astype("float32")
    global rnn_train
    global step
    if extras["train"]:
        step += 1
        if step > switch_step:
            rnn_train = True
        if both or not rnn_train:
            feed = {vs.vqvae_inputs: x,
                    vs.bn_flag: 0.}
            outs = [vs.vqvae_rec_loss, vs.vqvae_loss, vs.vqvae_train_step, vs.z_i_x]
            r = sess.run(outs, feed_dict=feed)
            vqvae_l = r[0]
            vqvae_t_l = r[1]
            vqvae_step = r[2]
        if rnn_train:
            feed = {vs.vqvae_inputs: x,
                    vs.bn_flag: 1.}
            outs = [vs.vqvae_rec_loss, vs.z_i_x]
            r = sess.run(outs, feed_dict=feed)
            vqvae_l = r[0]
            vqvae_t_l = r[1]

        discrete_z = r[-1]
        #discrete_z[3][:, 2:-2] == discrete_z[4][:, 1:-3]
        #discrete_z = discrete_z[:, :, 1:-2]
        shp = discrete_z.shape
        # always start with 0
        rnn_inputs = np.zeros((shp[2] + 1, shp[0], shp[1]))
        rnn_inputs[1:] = discrete_z.transpose(2, 0, 1) + 1.

        if both or rnn_train:
            feed = {vs.rnn_inputs: rnn_inputs,
                    vs.init_hidden: init_h,
                    vs.init_cell: init_c,
                    vs.init_q_hidden: init_q_h,
                    vs.init_q_cell: init_q_c}
            outs = [vs.rnn_rec_loss, vs.rnn_loss, vs.rnn_train_step]
            r = sess.run(outs, feed_dict=feed)
            rnn_l = r[0]
            rnn_t_l = r[1]
            rnn_step = r[2]
        if not rnn_train:
            feed = {vs.rnn_inputs: rnn_inputs,
                    vs.init_hidden: init_h,
                    vs.init_cell: init_c,
                    vs.init_q_hidden: init_q_h,
                    vs.init_q_cell: init_q_c}
            outs = [vs.rnn_rec_loss]
            r = sess.run(outs, feed_dict=feed)
            rnn_l = r[0]
    else:
        feed = {vs.vqvae_inputs: x,
                vs.bn_flag: 1.}
        outs = [vs.vqvae_rec_loss, vs.z_i_x]
        r = sess.run(outs, feed_dict=feed)
        vqvae_l = r[0]

        discrete_z = r[-1]
        #discrete_z = discrete_z[:, :, 1:-2]
        shp = discrete_z.shape
        # always start with 0
        rnn_inputs = np.zeros((shp[2] + 1, shp[0], shp[1]))
        rnn_inputs[1:] = discrete_z.transpose(2, 0, 1) + 1.

        feed = {vs.rnn_inputs: rnn_inputs,
                vs.init_hidden: init_h,
                vs.init_cell: init_c,
                vs.init_q_hidden: init_q_h,
                vs.init_q_cell: init_q_c}
        outs = [vs.rnn_rec_loss]
        r = sess.run(outs, feed_dict=feed)
        rnn_l = r[0]
    return [vqvae_l, rnn_l], None, stateful_args

with tf.Session(graph=g) as sess:
    run_loop(sess,
             loop, train_itr,
             loop, valid_itr,
             n_steps=75000,
             n_train_steps_per=5000,
             n_valid_steps_per=500)
