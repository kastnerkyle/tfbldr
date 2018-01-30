from __future__ import print_function
import tensorflow as tf
import numpy as np
import uuid
from scipy import linalg
import tensorflow as tf
import shutil
import socket
import os
import re
import copy
import sys
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import logging
from collections import OrderedDict

logging.basicConfig(level=logging.INFO,
                    format='%(message)s')
logger = logging.getLogger(__name__)

string_f = StringIO()
ch = logging.StreamHandler(string_f)
# Automatically put the HTML break characters on there for html logger
formatter = logging.Formatter('%(message)s<br>')
ch.setFormatter(formatter)
logger.addHandler(ch)

def fet_logger():
    return logger

sys.setrecursionlimit(40000)

# Storage of internal shared
_lib_shared_params = OrderedDict()


def _get_name():
    return str(uuid.uuid4())


def _get_shared(name):
    if name in _lib_shared_params.keys():
        logger.info("Found name %s in shared parameters" % name)
        return _lib_shared_params[name]
    else:
        raise NameError("Name not found in shared params!")


def _set_shared(name, variable):
    if name in _lib_shared_params.keys():
        raise ValueError("Trying to set key %s which already exists!" % name)
    _lib_shared_params[name] = variable


def get_params_dict():
    return _lib_shared_params


def print_network(params_dict):
    logger.info("=====================")
    logger.info("Model")
    logger.info(" ")
    logger.info("---------------------")
    for k, v in params_dict.items():
        strip_name = "_".join(k.split("_")[1:])
        k_count = np.prod(shape(v)) / float(1E3)
        logger.info("{}: {}K".format(strip_name, k_count))
    params = params_dict.values()
    n_params = sum([np.prod(shape(p)) for p in params])
    logger.info("---------------------")
    logger.info(" ")
    logger.info("Total: {}M".format(n_params / float(1E6)))
    logger.info("=====================")


def shape(x):
    r = x.get_shape().as_list()
    r = [ri if ri != None else -1 for ri in r]

    if len([ri for ri in r if ri == -1]) > 1:
        raise ValueError("Too many None shapes in shape dim {}, should only 1 -1 dim at most".format(r))
    return r


def ndim(x):
    return len(shape(x))


def np_zeros(shape):
    """
    Builds a numpy variable filled with zeros
    Parameters
    ----------
    shape, tuple of ints
        shape of zeros to initialize
    Returns
    -------
    initialized_zeros, array-like
        Array-like of zeros the same size as shape parameter
    """
    return np.zeros(shape).astype("float32")


def np_normal(shape, random_state, scale=0.01):
    """
    Builds a numpy variable filled with normal random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 0.01)
        default of 0.01 results in normal random values with variance 0.01
    Returns
    -------
    initialized_normal, array-like
        Array-like of normal random values the same size as shape parameter
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
    else:
        shp = shape
    return (scale * random_state.randn(*shp)).astype("float32")


def np_tanh_fan_normal(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 1.)
        default of 1. results in normal random values
        with sqrt(2 / (fan in + fan out)) scale
    Returns
    -------
    initialized_fan, array-like
        Array-like of random values the same size as shape parameter
    References
    ----------
    Understanding the difficulty of training deep feedforward neural networks
        X. Glorot, Y. Bengio
    """
    # The . after the 2 is critical! shape has dtype int...
    if type(shape[0]) is tuple:
        kern_sum = np.prod(shape[0]) + np.prod(shape[1])
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
    else:
        kern_sum = np.sum(shape)
        shp = shape
    var = scale * np.sqrt(2. / kern_sum)
    return var * random_state.randn(*shp).astype("float32")


def np_variance_scaled_uniform(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 1.)
        default of 1. results in uniform random values
        with 1 * sqrt(1 / (n_dims)) scale
    Returns
    -------
    initialized_scaled, array-like
        Array-like of random values the same size as shape parameter
    References
    ----------
    Efficient Backprop
        Y. LeCun, L. Bottou, G. Orr, K. Muller
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
        kern_sum = np.prod(shape[0])
    else:
        shp = shape
        kern_sum = shape[0]
    #  Make sure bounds aren't the same
    bound = scale * np.sqrt(3. / kern_sum)  # sqrt(3) for std of uniform
    return random_state.uniform(low=-bound, high=bound, size=shp).astype(
        "float32")


def np_ortho(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with orthonormal random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 1.)
        default of 1. results in orthonormal random values sacled by 1.
    Returns
    -------
    initialized_ortho, array-like
        Array-like of random values the same size as shape parameter
    References
    ----------
    Exact solutions to the nonlinear dynamics of learning in deep linear
    neural networks
        A. Saxe, J. McClelland, S. Ganguli
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
        flat_shp = (shp[0], np.prd(shp[1:]))
    else:
        shp = shape
        flat_shp = shape
    g = random_state.randn(*flat_shp)
    U, S, VT = linalg.svd(g, full_matrices=False)
    res = U if U.shape == flat_shp else VT  # pick one with the correct shape
    res = res.reshape(shp)
    return (scale * res).astype("float32")


def make_numpy_biases(bias_dims):
    return [np_zeros((dim,)) for dim in bias_dims]


def make_numpy_weights(in_dim, out_dims, random_state, init=None,
                       scale="default"):
    """
    Will return as many things as are in the list of out_dims
    You *must* get a list back, even for 1 element retuTrue
    blah, = make_weights(...)
    or
    [blah] = make_weights(...)
    """
    ff = [None] * len(out_dims)
    for i, out_dim in enumerate(out_dims):
        if init is None:
            if in_dim == out_dim:
                ff[i] = np_ortho
            else:
                ff[i] = np_variance_scaled_uniform
        else:
            raise ValueError("Unknown init type %s" % init)
    if scale == "default":
        ws = [ff[i]((in_dim, out_dim), random_state)
              for i, out_dim in enumerate(out_dims)]
    else:
        ws = [ff[i]((in_dim, out_dim), random_state, scale=scale)
              for i, out_dim in enumerate(out_dims)]
    return ws


def dot(a, b):
    # Generalized dot for nd sequences, assumes last axis is projection
    # b must be rank 2
    # rediculously hacky string parsing... wowie
    a_tup = shape(a)
    b_tup = shape(b)
    a_i = tf.reshape(a, [-1, a_tup[-1]])
    a_n = tf.matmul(a_i, b)
    a_n = tf.reshape(a_n, list(a_tup[:-1]) + [b_tup[-1]])
    return a_n


def scan(fn, sequences, outputs_info):
    # for some reason TF step needs initializer passed as first argument?
    # a tiny wrapper to tf.scan to make my life easier
    # closer to theano scan, allows for step functions with multiple arguments
    # may eventually have kwargs which match theano
    for i in range(len(sequences)):
        # Try to accomodate for masks...
        seq = sequences[i]
        nd = ndim(seq)
        if nd == 3:
            pass
        elif nd < 3:
            sequences[i] = tf.expand_dims(sequences[i], nd)
        else:
            raise ValueError("Ndim too different to correct")

    def check(l):
        shapes = [shape(s) for s in l]
        # for now assume -1, can add axis argument later
        # check shapes match for concatenation
        compat = [ls for ls in shapes if ls[:-1] == shapes[0][:-1]]
        if len(compat) != len(shapes):
            raise ValueError("Tensors *must* be the same dim for now")

    check(sequences)
    check(outputs_info)

    seqs_shapes = [shape(s) for s in sequences]
    nd = len(seqs_shapes[0])
    seq_pack = tf.concat(sequences, nd - 1)
    outs_shapes = [shape(o) for o in outputs_info]
    nd = len(outs_shapes[0])
    init_pack = tf.concat(outputs_info, nd - 1)

    assert len(shape(seq_pack)) == 3
    assert len(shape(init_pack)) == 2

    def s_e(shps):
        starts = []
        ends = []
        prev_shp = 0
        for n, shp in enumerate(shps):
            start = prev_shp
            end = start + shp[-1]
            starts.append(start)
            ends.append(end)
            prev_shp = end
        return starts, ends

    # TF puts the initializer in front?
    def fnwrap(initializer, elems):
        starts, ends = s_e(seqs_shapes)
        sliced_elems = [elems[:, start:end] for start, end in zip(starts, ends)]
        starts, ends = s_e(outs_shapes)
        sliced_inits = [initializer[:, start:end]
                        for start, end in zip(starts, ends)]
        t = []
        t.extend(sliced_elems)
        t.extend(sliced_inits)
        # elems first then inits
        outs = fn(*t)
        nd = len(outs_shapes[0])
        outs_pack = tf.concat(outs, nd - 1)
        return outs_pack

    r = tf.scan(fnwrap, seq_pack, initializer=init_pack)

    if len(outs_shapes) > 1:
        starts, ends = s_e(outs_shapes)
        o = [r[:, :, start:end] for start, end in zip(starts, ends)]
        return o
    else:
        return r


def Linear(list_of_inputs, input_dims, output_dim, random_state, name=None,
           init=None, scale="default", weight_norm=None, biases=True):
    """
    Can pass weights and biases directly if needed through init
    """
    if weight_norm is None:
        # Let other classes delegate to default of linear
        weight_norm = True
    # assume both have same shape ...
    nd = ndim(list_of_inputs[0])
    input_var = tf.concat(list_of_inputs, axis=nd - 1)
    input_dim = sum(input_dims)
    terms = []
    if (init is None) or (type(init) is str):
        weight_values, = make_numpy_weights(input_dim, [output_dim],
                                            random_state=random_state,
                                            init=init, scale=scale)
    else:
        weight_values = init[0]

    if name is None:
        name = _get_name()
    elif name[0] is None:
        name = (_get_name(),) + name[1:]
        name = "_".join(name)

    name_w = name + "_linear_w"
    name_b = name + "_linear_b"
    name_wn = name + "_linear_wn"

    try:
        weight = _get_shared(name_w)
    except NameError:
        weight = tf.Variable(weight_values, trainable=True)
        _set_shared(name_w, weight)

    # Weight normalization... Kingma and Salimans
    # http://arxiv.org/abs/1602.07868
    if weight_norm:
        norm_values = np.linalg.norm(weight_values, axis=0)
        try:
            norms = _get_shared(name_wn)
        except NameError:
            norms = tf.Variable(norm_values, trainable=True)
            _set_shared(name_wn, norms)
        norm = tf.sqrt(tf.reduce_sum(tf.abs(weight ** 2), reduction_indices=[0],
                                     keep_dims=True))
        normed_weight = weight * (norms / norm)
        terms.append(dot(input_var, normed_weight))
    else:
        terms.append(dot(input_var, weight))

    if biases:
        if (init is None) or (type(init) is str):
            b, = make_numpy_biases([output_dim])
        else:
            b = init[1]
        try:
            biases = _get_shared(name_b)
        except NameError:
            biases = tf.Variable(b, trainable=True)
            _set_shared(name_b, biases)
        terms.append(biases)
    out = reduce(lambda a, b: a + b, terms)
    return out
