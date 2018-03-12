from tfbldr.datasets import fetch_mnist
from tfbldr.nodes import Conv2d
from tfbldr.nodes import ConvTranspose2d
from tfbldr.nodes import VqEmbedding
from tfbldr.nodes import BatchNorm2d
from tfbldr.nodes import ReLU
from tfbldr.nodes import Sigmoid
from tfbldr.nodes import BernoulliCrossEntropyCost
from tfbldr import dot
from tfbldr import get_params_dict
import tensorflow as tf
import numpy as np

"""
mnist = fetch_mnist()
"""
images = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
bn_flag = tf.placeholder_with_default(tf.zeros(shape=[]), shape=[])

random_state = np.random.RandomState(1999)

l1_dim = (16, 4, 4, 2)
l2_dim = (32, 4, 4, 2)
l3_dim = (64, 1, 1, 1)
l_dims = [l1_dim, l2_dim, l3_dim]
#bpad = [1, 1, 1, 1]
stride_div = np.prod([ld[-1] for ld in l_dims])

latent = tf.placeholder(tf.float32, shape=[None, 28 // stride_div, 28 // stride_div, l_dims[-1][0]])

bpad = 1

def create_encoder(inp):
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


def create_decoder(latent):
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


def create_model(inp):
    z_e_x = create_encoder(inp)
    z_q_x, emb = VqEmbedding(z_e_x, 64, 512, random_state=random_state, name="embed")
    """
    embedding_weight = random_state.randn(512, 64).astype(np.float32)
    emb = tf.Variable(embedding_weight, trainable=True)
    emb_r = tf.transpose(emb, (1, 0))
    sq_diff = tf.square(z_e_x[..., None] - emb_r[None, None, None])
    sum_sq_diff = tf.reduce_sum(sq_diff, axis=-2)
    discrete_latent_idx = tf.reduce_min(sum_sq_diff, axis=-1)
    shp = _shape(discrete_latent_idx)
    flat_idx = tf.cast(tf.reshape(discrete_latent_idx, (-1,)), tf.int32)
    lu_vectors = tf.nn.embedding_lookup(emb, flat_idx)
    shp2 = _shape(lu_vectors)
    z_q_x = tf.reshape(lu_vectors, (-1, shp[1], shp[2], shp2[-1]))
    """
    x_tilde = create_decoder(z_q_x)
    return x_tilde, z_e_x, z_q_x

x_tilde, z_e_x, z_q_x = create_model(images)
rec_loss = tf.reduce_mean(BernoulliCrossEntropyCost(x_tilde, images))
vq_loss = tf.reduce_mean(tf.square(tf.stop_gradient(z_e_x) - z_q_x))
commit_loss = tf.reduce_mean(tf.square(z_e_x - tf.stop_gradient(z_q_x)))
beta = 0.25
loss = rec_loss + vq_loss + beta * commit_loss
params = get_params_dict()

enc_params = [params[k] for k in params.keys() if "enc" in k]
dec_params = [params[k] for k in params.keys() if "dec" in k]
emb_params = [params[k] for k in params.keys() if "embed" in k]

decoder_grads = list(zip(tf.gradients(loss, dec_params), dec_params))
embed_grads = list(zip(tf.gradients(vq_loss, emb_params), emb_params))
grad_z = tf.gradients(rec_loss, z_q_x)
enc_grads = [(tf.gradients(z_e_x, p, grad_z)[0] + beta * tf.gradients(commit_loss, p)[0], p) for p in enc_params]

from IPython import embed; embed(); raise ValueError()
