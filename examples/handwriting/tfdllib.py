from __future__ import print_function
import tensorflow as tf
import numpy as np
import uuid
from scipy import linalg
from scipy.stats import truncnorm
from scipy.misc import factorial
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

weight_norm_default = False
def get_weight_norm_default():
    return weight_norm_default


def print_network(params_dict):
    logger.info("=====================")
    logger.info("Model Summary")
    logger.info("format: {name} {shape}, {parameter_count}")
    logger.info("---------------------")
    for k, v in params_dict.items():
        #strip_name = "_".join(k.split("_")[1:])
        strip_name = k
        shp = tuple(shape(v))
        k_count = np.prod(shp) / float(1E3)
        logger.info("{} {}, {}K".format(strip_name, shp, k_count))
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


def sigmoid(x):
    return tf.sigmoid(x)


def tanh(x):
    return tf.tanh(x)


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


def np_truncated_normal(shape, random_state, scale=0.075):
    """
    Builds a numpy variable filled with truncated normal random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 0.075)
        default of 0.075
    Returns
    -------
    initialized_normal, array-like
        Array-like of truncated normal random values the same size as shape parameter
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
    else:
        shp = shape

    sigma = scale
    lower = -2 * sigma
    upper = 2 * sigma
    mu = 0
    N = np.prod(shp)
    samples = truncnorm.rvs(
              (lower - mu) / sigma, (upper - mu) / sigma,
              loc=mu, scale=sigma, size=N, random_state=random_state)
    return samples.reshape(shp).astype("float32")


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
        elif init == "normal":
            ff[i] = np_normal
        elif init == "truncated_normal":
            ff[i] = np_truncated_normal
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
        shapes = [shape(s) for s in l if s is not None]
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
    outs_shapes = [shape(o) for o in outputs_info if o is not None]
    nd = len(outs_shapes[0])
    init_pack = tf.concat([o for o in outputs_info if o is not None], nd - 1)

    if None in outputs_info:
        raise ValueError("None in outputs_info not currently supported")

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


def Linear(list_of_inputs, list_of_input_dims, output_dim, random_state, name=None,
           init=None, scale="default", weight_norm=None, biases=True, strict=True):
    """
    Can pass weights and biases directly if needed through init
    """
    if weight_norm is None:
        # Let other classes delegate to default of linear
        weight_norm = get_weight_norm_default()
    # assume both have same shape ...
    nd = ndim(list_of_inputs[0])
    input_var = tf.concat(list_of_inputs, axis=nd - 1)
    input_dim = sum(list_of_input_dims)
    terms = []
    if (init is None) or (type(init) is str):
        weight_values, = make_numpy_weights(input_dim, [output_dim],
                                            random_state=random_state,
                                            init=init, scale=scale)
    else:
        weight_values = init[0]

    if name is None:
        name = _get_name()

    name_w = name + "_linear_w"
    name_b = name + "_linear_b"
    name_wn = name + "_linear_wn"
    name_out = name + "_linear_out"
    if strict:
        cur_defs = get_params_dict()
        if name_w in cur_defs:
            raise ValueError("Name {} already created in params dict!".format(name_w))

        if name_b in cur_defs:
            raise ValueError("Name {} already created in params dict!".format(name_b))

        if name_wn in cur_defs:
            raise ValueError("Name {} already created in params dict!".format(name_wn))

    try:
        weight = _get_shared(name_w)
    except NameError:
        weight = tf.Variable(weight_values, trainable=True, name=name_w)
        _set_shared(name_w, weight)

    # Weight normalization... Kingma and Salimans
    # http://arxiv.org/abs/1602.07868
    if weight_norm:
        norm_values = np.linalg.norm(weight_values, axis=0)
        try:
            norms = _get_shared(name_wn)
        except NameError:
            norms = tf.Variable(norm_values, trainable=True, name=name_wn)
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
            biases = tf.Variable(b, trainable=True, name=name_b)
            _set_shared(name_b, biases)
        terms.append(biases)
    out = reduce(lambda a, b: a + b, terms)
    out = tf.identity(out, name=name_out)
    return out


def gru_weights(input_dim, hidden_dim, forward_init=None, hidden_init="normal",
                random_state=None):
    if random_state is None:
        raise ValueError("Must pass random_state!")
    shape = (input_dim, hidden_dim)
    if forward_init == "normal":
        W = np.hstack([np_normal(shape, random_state),
                       np_normal(shape, random_state),
                       np_normal(shape, random_state)])
    elif forward_init == "fan":
        W = np.hstack([np_tanh_fan_normal(shape, random_state),
                       np_tanh_fan_normal(shape, random_state),
                       np_tanh_fan_normal(shape, random_state)])
    elif forward_init == "truncated_normal":
        W = np.hstack([np_truncated_normal(shape, random_state),
                       np_truncated_normal(shape, random_state),
                       np_truncated_normal(shape, random_state)])
    elif forward_init is None:
        if input_dim == hidden_dim:
            W = np.hstack([np_ortho(shape, random_state),
                           np_ortho(shape, random_state),
                           np_ortho(shape, random_state)])
        else:
            # lecun
            W = np.hstack([np_variance_scaled_uniform(shape, random_state),
                           np_variance_scaled_uniform(shape, random_state),
                           np_variance_scaled_uniform(shape, random_state)])
    else:
        raise ValueError("Unknown forward init type %s" % forward_init)
    b = np_zeros((3 * shape[1],))

    if hidden_init == "normal":
        Wur = np.hstack([np_normal((shape[1], shape[1]), random_state),
                         np_normal((shape[1], shape[1]), random_state), ])
        U = np_normal((shape[1], shape[1]), random_state)
    elif hidden_init == "ortho":
        Wur = np.hstack([np_ortho((shape[1], shape[1]), random_state),
                         np_ortho((shape[1], shape[1]), random_state), ])
        U = np_ortho((shape[1], shape[1]), random_state)
    elif hidden_init == "truncated_normal":
        Wur = np.hstack([np_truncated_normal((shape[1], shape[1]), random_state),
                         np_truncated_normal((shape[1], shape[1]), random_state), ])
        U = np_ortho((shape[1], shape[1]), random_state)
    return W, b, Wur, U


def GRU(inp, gate_inp, previous_state, input_dim, hidden_dim, random_state,
        mask=None, name=None, init=None, scale="default", weight_norm=None,
        biases=False):
        if init is None:
            hidden_init = "ortho"
        elif init == "ortho":
            hidden_init = "ortho"
        elif init == "normal":
            hidden_init = "normal"
        elif init == "truncated_normal":
            hidden_init = "truncated_normal"
        else:
            raise ValueError("Not yet configured for other inits")

        ndi = ndim(inp)
        if mask is None:
            if ndi == 2:
                mask = tf.ones_like(inp)
            else:
                raise ValueError("Unhandled ndim")

        ndm = ndim(mask)
        if ndm == (ndi - 1):
            mask = tf.expand_dims(mask, ndm - 1)

        if hasattr(input_dim, "__len__"):
            input_dim = input_dim[0]

        if hasattr(hidden_dim, "__len__"):
            hidden_dim = hidden_dim[0]

        _, _, Wur, U = gru_weights(input_dim, hidden_dim,
                                   hidden_init=hidden_init,
                                   random_state=random_state)
        if name is None:
            name = _get_name()

        dim = hidden_dim
        f1 = Linear([previous_state], [2 * hidden_dim], 2 * hidden_dim,
                    random_state, name=name +"_update/reset", init=[Wur],
                    biases=biases, weight_norm=weight_norm)
        gates = sigmoid(f1 + gate_inp)
        update = gates[:, :dim]
        reset = gates[:, dim:]
        state_reset = previous_state * reset
        f2 = Linear([state_reset], [hidden_dim], hidden_dim,
                    random_state, name=name + "_state", init=[U], biases=biases,
                    weight_norm=weight_norm)
        next_state = tf.tanh(f2 + inp)
        next_state = next_state * update + previous_state * (1. - update)
        next_state = mask * next_state + (1. - mask) * previous_state
        next_state = tf.identity(next_state, name=name + "_next_state")
        return next_state


def GRUFork(list_of_inputs, list_of_input_dims, output_dim, random_state, name=None,
            init=None, scale="default", weight_norm=None, biases=True):
        if name is None:
            name = _get_name()
        gates = Linear(list_of_inputs, list_of_input_dims, 3 * output_dim,
                       random_state=random_state,
                       name=name + "_gates", init=init, scale=scale,
                       weight_norm=weight_norm, biases=biases)
        dim = output_dim
        nd = ndim(gates)
        if nd == 2:
            d = gates[:, :dim]
            g = gates[:, dim:]
        elif nd == 3:
            d = gates[:, :, :dim]
            g = gates[:, :, dim:]
        else:
            raise ValueError("Unsupported ndim")
        return d, g


def _slice_state(tensor, hidden_dim):
    """
    Used to slice the final state of GRU, LSTM to the suitable part
    """
    part = 0
    if len(shape(tensor)) == 2:
        return tensor[:, part * hidden_dim:(part + 1) * hidden_dim]
    elif len(shape(tensor)) == 3:
        return tensor[:, :, part * hidden_dim:(part + 1) * hidden_dim]
    else:
        raise ValueError("Unknown dim")


def GaussianAttention(list_of_step_inputs, list_of_step_input_dims,
                      previous_state,
                      previous_attention_position,
                      previous_attention_weight,
                      full_conditioning_tensor,
                      previous_attention_weight_dim,
                      next_proj_dim,
                      att_dim=10,
                      average_step=1.,
                      min_step=0.,
                      max_step=None,
                      cell_type="gru",
                      step_mask=None, conditioning_mask=None, name=None,
                      weight_norm=None,
                      random_state=None, strict=True, init="default"):
    """
    returns h_t (hidden state of inner rnn)
            k_t (attention position for each attention element)
            w_t (attention weights for each element of conditioning tensor)
        Use w_t for following projection/prediction
    """
    if name is None:
        name = _get_name()
    name = name + "_gaussian_attention"

    #TODO: specialize for jose style init...
    if init == "default":
        forward_init = None
        hidden_init = "ortho"
    elif init == "normal":
        forward_init = "normal"
        hidden_init = "normal"
    elif init == "truncated_normal":
        forward_init = "truncated_normal"
        hidden_init = "truncated_normal"
    else:
        raise ValueError("Unknown init value {}".format(init))

    check = any([len(shape(si)) != 2 for si in list_of_step_inputs])
    if check:
        raise ValueError("Unable to support step_input with n_dims != 2")

    if cell_type == "gru":
        # INIT FUNC
        fork_inp, fork_inp_gate = GRUFork(list_of_step_inputs + [previous_attention_weight],
                                          list_of_step_input_dims + [previous_attention_weight_dim],
                                          next_proj_dim,
                                          name=name + "_fork", random_state=random_state,
                                          init=forward_init
                                          )
        h_t = GRU(fork_inp, fork_inp_gate, previous_state, [next_proj_dim], next_proj_dim,
                  mask=step_mask, name=name + "_rec", random_state=random_state,
                  init=hidden_init)
    elif cell_type == "lstm":
        raise ValueError("LSTM COND LOGIC NYI")
        """
        fork1 = lstm_fork(list_of_step_inputs + [previous_attention_weight],
                        list_of_step_input_dims + [conditioning_dim], next_proj_dim,
                        name=name + "_fork", random_state=random_state,
                        init_func="normal")
        h_t = lstm(fork1, previous_state, [next_proj_dim], next_proj_dim,
                   mask=step_mask, name=name + "_rec", random_state=random_state,
                   init_func="normal")
        """
    else:
        raise ValueError("Unsupported cell_type %s" % cell_type)

    # tf.shape and tensor.shape are not the same...
    u = tf.cast(tf.range(0., tf.shape(full_conditioning_tensor)[0]), dtype=tf.float32)
    u = tf.expand_dims(tf.expand_dims(u, axis=0), axis=0)

    h_sub_t = _slice_state(h_t, next_proj_dim)
    ret = Linear(
        list_of_inputs=[h_sub_t], list_of_input_dims=[next_proj_dim],
        output_dim=3 * att_dim, name=name + "_group", weight_norm=weight_norm,
        random_state=random_state,
        strict=strict, init=forward_init)
    a_t = ret[:, :att_dim]
    b_t = ret[:, att_dim:2 * att_dim]
    k_t = ret[:, 2 * att_dim:]

    k_tm1 = previous_attention_position
    ctx = full_conditioning_tensor
    ctx_mask = conditioning_mask
    if ctx_mask is None:
        ctx_mask = 0. * ctx[:, :, 0] + 1.

    a_t = tf.exp(a_t)
    b_t = tf.exp(b_t)
    a_t = tf.identity(a_t, name=name + "_a_scale")
    b_t = tf.identity(b_t, name=name + "_b_scale")
    step_size = average_step * tf.exp(k_t)
    """
    if max_step is None:
        max_step = tensor.cast(ctx.shape[0], "float32")
    else:
        max_step = np.cast["float32"](float(max_step))
    step_size = step_size.clip(min_step, max_step)
    """
    k_t = k_tm1 + step_size
    k_t = tf.identity(k_t, name=name + "_position")
    # Don't let the gaussian go off the end
    #k_t = k_t.clip(np.cast["float32"](0.), tensor.cast(ctx.shape[0], "float32"))

    def calc_phi(lk_t, la_t, lb_t, lu):
        la_t = tf.expand_dims(la_t, axis=2)
        lb_t = tf.expand_dims(lb_t, axis=2)
        lk_t = tf.expand_dims(lk_t, axis=2)
        ss1 = (lk_t - lu) ** 2
        ss2 = -lb_t * ss1
        ss3 = la_t * tf.exp(ss2)
        ss4 = tf.reduce_sum(ss3, axis=1)
        return ss4

    ss_t = calc_phi(k_t, a_t, b_t, u)
    ss_t = tf.identity(ss_t, name=name + "_phi")

    # calculate and return stopping criteria
    #sh_t = calc_phi(k_t, a_t, b_t, u_max)
    ss5 = tf.expand_dims(ss_t, axis=2)
    # mask using conditioning mask
    ss6 = ss5 * tf.transpose(ctx, (1, 0, 2)) * tf.transpose(tf.expand_dims(ctx_mask, axis=2), (1, 0, 2))
    w_t = tf.reduce_sum(ss6, axis=1)
    w_t = tf.identity(w_t, name=name + "_post_weighting")
    return h_t, k_t, w_t, ss_t


def LogBernoulliAndCorrelatedLogGMM(
        list_of_inputs, list_of_input_dims, output_dim=2, name=None, n_components=5,
        weight_norm=None, random_state=None, strict=True,
        init=None):
    """
    returns log_bernoulli, coeffs, mus, log_sigmas, corr
    """
    assert n_components >= 1
    if name is None:
        name = _get_name()
    else:
        name = name + "_bernoulli_and_correlated_log_gaussian_mixture"


    def _reshape(l, d=n_components):
        if d == 1:
            shp = shape(l)
            t = tf.reshape(l, shp[:-1] + [1, shp[-1]])
            return t
        if len(shape(l)) == 2:
            t = tf.reshape(l, (-1, output_dim, d))
        elif len(shape(l)) == 3:
            shp = shape(l)
            t = tf.reshape(l, (-1, shp[1], output_dim, d))
        else:
            raise ValueError("input ndim not supported for gaussian "
                             "mixture layer")
        return t

    if output_dim != 2:
        raise ValueError("General calculation for GMM not yet implemented")

    mus = Linear(
        list_of_inputs=list_of_inputs, list_of_input_dims=list_of_input_dims,
        output_dim=n_components * output_dim, name=name + "_mus_pre",
        weight_norm=weight_norm, random_state=random_state,
        strict=strict, init=init)
    mus = _reshape(mus)
    mus = tf.identity(mus, name=name + "_mus")

    log_sigmas = Linear(
        list_of_inputs=list_of_inputs, list_of_input_dims=list_of_input_dims,
        output_dim=n_components * output_dim, name=name + "_log_sigmas_pre",
        weight_norm=weight_norm, random_state=random_state,
        strict=strict, init=init)
    log_sigmas = _reshape(log_sigmas)
    log_sigmas = tf.identity(log_sigmas, name=name + "_log_sigmas")

    coeffs = Linear(
        list_of_inputs=list_of_inputs, list_of_input_dims=list_of_input_dims,
        output_dim=n_components, name=name + "_coeffs_pre",
        weight_norm=weight_norm, random_state=random_state,
        strict=strict, init=init)
    coeffs = tf.nn.softmax(coeffs)
    coeffs = _reshape(coeffs, 1)
    coeffs = tf.identity(coeffs, name=name + "_coeffs")

    calc_corr = int(factorial(output_dim ** 2 // 2 - 1))
    corrs = Linear(
        list_of_inputs=list_of_inputs, list_of_input_dims=list_of_input_dims,
        output_dim=n_components * calc_corr, name=name + "_corrs_pre",
        weight_norm=weight_norm, random_state=random_state,
        strict=strict, init=init)
    corrs = tf.tanh(corrs)
    corrs = _reshape(corrs, calc_corr)
    corrs = tf.identity(corrs, name + "_corrs")

    log_bernoullis = Linear(
        list_of_inputs=list_of_inputs, list_of_input_dims=list_of_input_dims,
        output_dim=1, name=name + "_log_bernoullis_pre",
        weight_norm=weight_norm, random_state=random_state,
        strict=strict, init=init)
    log_bernoullis = tf.identity(log_bernoullis, name + "_log_bernoullis")
    return log_bernoullis, coeffs, mus, log_sigmas, corrs


# from A d B
# https://github.com/adbrebs/handwriting/blob/master/model.py
def _logsumexp(inputs, axis=-1):
    max_i = tf.reduce_max(inputs, axis=axis)
    z = tf.log(tf.reduce_sum(tf.exp(inputs - max_i[..., None]), axis=axis)) + max_i
    return z


def LogBernoulliAndCorrelatedLogGMMCost(
    log_bernoulli_values, coeff_values, mu_values, log_sigma_values, corr_values,
    true_values, name=None):
    """
    Log bernoulli combined with correlated gaussian mixture model negative log
    likelihood compared to true_values.

    This is typically paired with LogBernoulliAndCorrelatedLogGMM

    Based on implementation from Junyoung Chung.

    Parameters
    ----------
    log_bernoulli_values : tensor, shape
        The predicted values out of some layer, normallu a linear layer
    coeff_values : tensor, shape
        The predicted values out of some layer, normally a softmax layer
    mu_values : tensor, shape
        The predicted values out of some layer, normally a linear layer
    log_sigma_values : tensor, shape
        The predicted values out of some layer, normally a linear layer
    true_values : tensor, shape[:-1]
        Ground truth values. Must be the same shape as mu_values.shape[:-1].
    Returns
    -------
    nll : tensor, shape predicted_values.shape[1:]
        The cost per sample, or per sample per step if 3D
    References
    ----------
    [1] University of Utah Lectures
        http://www.cs.utah.edu/~piyush/teaching/gmm.pdf
    [2] Statlect.com
        http://www.statlect.com/normal_distribution_maximum_likelihood.htm
    """
    if name == None:
        name = _get_name()
    else:
        name = name

    tv = true_values
    if len(shape(tv)) == 3:
        true_values = tf.expand_dims(tv, axis=2)
    elif len(shape(tv)) == 2:
        true_values = tf.expand_dims(tv, axis=1)
    else:
        raise ValueError("shape of labels not currently supported {}".format(shape(tv)))

    def _subslice(arr, idx):
        if len(shape(arr)) == 3:
            return arr[:, idx]
        elif len(shape(arr)) == 4:
            return arr[:, :, idx]
        raise ValueError("Unsupported ndim {}".format(shape(arr)))

    mu_1 = _subslice(mu_values, 0)
    mu_2 = _subslice(mu_values, 1)

    log_sigma_1 = _subslice(log_sigma_values, 0)
    log_sigma_2 = _subslice(log_sigma_values, 1)

    true_0 = true_values[..., 0]
    true_1 = true_values[..., 1]
    true_2 = true_values[..., 2]

    # thanks to DWF
    a, t = log_bernoulli_values, true_0
    c_b = -1. * tf.reduce_sum(t * tf.nn.softplus(-a) + (1. - t) * tf.nn.softplus(a), axis=len(shape(t)) - 1)
    c_b = tf.identity(c_b, name=name + "_binary_nll")

    corr_values = _subslice(corr_values, 0)
    coeff_values = _subslice(coeff_values, 0)

    buff = 1 - corr_values ** 2 + 1E-8
    inner1 = (0.5 * tf.log(buff) +
              log_sigma_1 + log_sigma_2 + tf.log(2 * np.pi))

    z1 = ((true_1 - mu_1) ** 2) / tf.exp(2 * log_sigma_1)
    z2 = ((true_2 - mu_2) ** 2) / tf.exp(2 * log_sigma_2)
    zr = (2 * corr_values * (true_1 - mu_1) * (true_2 - mu_2)) / (
        tf.exp(log_sigma_1 + log_sigma_2))
    z = z1 + z2 - zr

    inner2 = .5 * (1. / buff)
    cost = -(inner1 + z * inner2)
    cost = tf.identity(cost, name=name + "_gaussian_nll")

    coeff_log = tf.log(coeff_values)
    coeff_log = tf.identity(coeff_log, name=name + "_coeff_entropy")

    nll = -_logsumexp(cost + coeff_log,
                      axis=len(shape(coeff_values)) - 1) - c_b
    nll = tf.identity(nll, name=name + "_full_nll")
    return nll
