from tfbldr.datasets import fetch_mnist
from tfbldr.nodes import Conv2d, BatchNorm2d, ReLU
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
#bpad = [1, 1, 1, 1]
bpad = 1

def create_model():
    l1 = Conv2d([images], [1], l1_dim[0], kernel_size=l1_dim[1:3], name="c1",
                strides=l1_dim[-1],
                border_mode=bpad,
                random_state=random_state)
    bn_l1 = BatchNorm2d(l1, bn_flag, name="bn_c1")
    r_l1 = ReLU(bn_l1)

    l2 = Conv2d([r_l1], [l1_dim[0]], l2_dim[0], kernel_size=l2_dim[1:3], name="c2",
                strides=l2_dim[-1],
                border_mode=bpad,
                random_state=random_state)
    bn_l2 = BatchNorm2d(l2, bn_flag, name="bn_c2")
    r_l2 = ReLU(bn_l2)

    l3 = Conv2d([r_l2], [l2_dim[0]], l3_dim[0], kernel_size=l3_dim[1:3], name="c3",
                random_state=random_state)
    bn_l3 = BatchNorm2d(l3, bn_flag, name="bn_c3")
    from IPython import embed; embed(); raise ValueError()

create_model()



