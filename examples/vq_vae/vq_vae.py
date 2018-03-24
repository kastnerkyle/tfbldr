from tfbldr.datasets import fetch_mnist
from tfbldr.nodes import Conv2d
from tfbldr.nodes import ConvTranspose2d
from tfbldr.nodes import VqEmbedding
from tfbldr.nodes import BatchNorm2d
from tfbldr.nodes import ReLU
from tfbldr.nodes import Sigmoid
from tfbldr.nodes import BernoulliCrossEntropyCost
from tfbldr.datasets import list_iterator
from tfbldr import get_params_dict
from tfbldr import run_loop
import tensorflow as tf
import numpy as np
from collections import namedtuple

mnist = fetch_mnist()
image_data = mnist["images"] / 255.
# save last 10k to validate on
train_image_data = image_data[:-10000]
val_image_data = image_data[-10000:]
train_itr_random_state = np.random.RandomState(1122)
val_itr_random_state = np.random.RandomState(1)
train_itr = list_iterator([train_image_data], 64, random_state=train_itr_random_state)
val_itr = list_iterator([val_image_data], 64, random_state=val_itr_random_state)

random_state = np.random.RandomState(1999)
l1_dim = (16, 4, 4, 2)
l2_dim = (32, 4, 4, 2)
l3_dim = (64, 1, 1, 1)
embedding_dim = 512
l_dims = [l1_dim, l2_dim, l3_dim]
stride_div = np.prod([ld[-1] for ld in l_dims])
bpad = 1

def create_encoder(inp, bn_flag):
    l1 = Conv2d([inp], [1], l_dims[0][0], kernel_size=l_dims[0][1:3], name="enc1",
                strides=l_dims[0][-1],
                border_mode=bpad,
                random_state=random_state)
    bn_l1 = BatchNorm2d(l1, bn_flag, name="bn_enc1")
    r_l1 = ReLU(bn_l1)

    l2 = Conv2d([r_l1], [l_dims[0][0]], l_dims[1][0], kernel_size=l_dims[1][1:3], name="enc2",
                strides=l_dims[1][-1],
                border_mode=bpad,
                random_state=random_state)
    bn_l2 = BatchNorm2d(l2, bn_flag, name="bn_enc2")
    r_l2 = ReLU(bn_l2)

    l3 = Conv2d([r_l2], [l_dims[1][0]], l_dims[2][0], kernel_size=l_dims[2][1:3], name="enc3",
                random_state=random_state)
    bn_l3 = BatchNorm2d(l3, bn_flag, name="bn_enc3")
    return bn_l3


def create_decoder(latent, bn_flag):
    l1 = Conv2d([latent], [l_dims[2][0]], l_dims[1][0], kernel_size=l_dims[2][1:3], name="dec1",
                random_state=random_state)
    bn_l1 = BatchNorm2d(l1, bn_flag, name="bn_dec3")
    r_l1 = ReLU(bn_l1)

    l2 = ConvTranspose2d([r_l1], [l_dims[1][0]], l_dims[0][0], kernel_size=l_dims[1][1:3], name="dec2",
                         strides=l_dims[1][-1],
                         border_mode=bpad,
                         random_state=random_state)
    bn_l2 = BatchNorm2d(l2, bn_flag, name="bn_dec2")
    r_l2 = ReLU(bn_l2)

    l3 = ConvTranspose2d([r_l2], [l_dims[0][0]], 1, kernel_size=l_dims[0][1:3], name="dec3",
                         strides=l_dims[0][-1],
                         border_mode=bpad,
                         random_state=random_state)
    s_l3 = Sigmoid(l3)
    return s_l3


def create_vqvae(inp, bn):
    z_e_x = create_encoder(inp, bn)
    z_st_q_x, z_i_x, z_nst_q_x, emb = VqEmbedding(z_e_x, l_dims[-1][0], embedding_dim, random_state=random_state, name="embed")
    z_q_x = z_nst_q_x
    x_tilde = create_decoder(z_q_x, bn)
    return x_tilde, z_e_x, z_q_x, z_i_x, emb

def create_graph():
    graph = tf.Graph()
    with graph.as_default():
        images = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        bn_flag = tf.placeholder_with_default(tf.zeros(shape=[]), shape=[])
        x_tilde, z_e_x, z_q_x, z_i_x, z_emb = create_vqvae(images, bn_flag)
        rec_loss = tf.reduce_mean(BernoulliCrossEntropyCost(x_tilde, images))
        vq_loss = tf.reduce_mean(tf.square(tf.stop_gradient(z_e_x) - z_q_x))
        commit_loss = tf.reduce_mean(tf.square(z_e_x - tf.stop_gradient(z_q_x)))
        #rec_loss = tf.reduce_mean(tf.reduce_sum(BernoulliCrossEntropyCost(x_tilde, images), axis=[1, 2]))
        #vq_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.stop_gradient(z_e_x) - z_q_x), axis=[1, 2, 3]))
        #commit_loss = tf.reduce_mean(tf.reduce_sum(tf.square(z_e_x - tf.stop_gradient(z_q_x)), axis=[1, 2, 3]))
        beta = 0.25
        loss = rec_loss + vq_loss + beta * commit_loss
        params = get_params_dict()

        enc_params = [params[k] for k in params.keys() if "enc" in k]
        dec_params = [params[k] for k in params.keys() if "dec" in k]
        emb_params = [params[k] for k in params.keys() if "embed" in k]

        dec_grads = list(zip(tf.gradients(loss, dec_params), dec_params))
        # scaled loss by alpha, but crank up vq loss grad
        # like having a higher lr only on embeds
        embed_grads = list(zip(tf.gradients(vq_loss, emb_params), emb_params))
        grad_z = tf.gradients(rec_loss, z_q_x)
        enc_grads = [(tf.gradients(z_e_x, p, grad_z)[0] + tf.gradients(beta * commit_loss, p)[0], p) for p in enc_params]

        learning_rate = 0.0002
        optimizer = tf.train.AdamOptimizer(learning_rate, use_locking=True)
        train_step = optimizer.apply_gradients(dec_grads + enc_grads + embed_grads)

    things_names = ["images",
                    "bn_flag",
                    "x_tilde",
                    "z_e_x",
                    "z_q_x",
                    "z_i_x",
                    "z_emb",
                    "loss",
                    "rec_loss",
                    "train_step"]
    things_tf = [images,
                 bn_flag,
                 x_tilde,
                 z_e_x,
                 z_q_x,
                 z_i_x,
                 z_emb,
                 loss,
                 rec_loss,
                 train_step]
    assert len(things_names) == len(things_tf)
    for tn, tt in zip(things_names, things_tf):
        graph.add_to_collection(tn, tt)
    train_model = namedtuple('Model', things_names)(*things_tf)
    return graph, train_model

g, vs = create_graph()

def loop(sess, itr, extras, stateful_args):
    x, = itr.next_batch()
    if extras["train"]:
        feed = {vs.images: x,
                vs.bn_flag: 0.}
        outs = [vs.rec_loss, vs.loss, vs.train_step]
        r = sess.run(outs, feed_dict=feed)
        l = r[0]
        t_l = r[1]
        step = r[2]
    else:
        feed = {vs.images: x,
                vs.bn_flag: 1.}
        outs = [vs.rec_loss]
        r = sess.run(outs, feed_dict=feed)
        l = r[0]
    return l, None, stateful_args

with tf.Session(graph=g) as sess:
    run_loop(sess,
             loop, train_itr,
             loop, val_itr,
             n_steps=10000,
             n_train_steps_per=1000,
             n_valid_steps_per=1000)
