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

def get_logger():
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

strict_mode_default = False
def get_strict_mode_default():
    return strict_mode_default


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

    #if len([ri for ri in r if ri == -1]) > 1:
    #    raise ValueError("Too many None shapes in shape dim {}, should only 1 -1 dim at most".format(r))
    return r


def ndim(x):
    return len(shape(x))


def dot(a, b):
    # Generalized dot for nd sequences, assumes last axis is projection
    # b must be rank 2
    a_tup = shape(a)
    b_tup = shape(b)
    if len(a_tup) == 2 and len(b_tup) == 2:
        return tf.matmul(a, b)
    elif len(a_tup) == 3 and len(b_tup) == 2:
        # more generic, supports multiple -1 axes
        return tf.einsum("ijk,kl->ijl", a, b)
        #a_i = tf.reshape(a, [-1, a_tup[-1]])
        #a_n = tf.matmul(a_i, b)
        #a_nf = tf.reshape(a_n, list(a_tup[:-1]) + [b_tup[-1]])
        #return a_nf
    else:
        raise ValueError("Shapes for arguments to dot() are {} and {}, not supported!".format(a_tup, b_tup))


def scan(fn, sequences, outputs_info):
    nonepos = [n for n, o in enumerate(outputs_info) if o is None]
    nonnone = [o for o in outputs_info if o is not None]
    sequences_and_nonnone = sequences + nonnone
    sliced = [s[0] for s in sequences] + nonnone
    inf_ret = fn(*sliced)
    if len(outputs_info) < len(inf_ret):
        raise ValueError("More outputs from `fn` than elements in outputs_info. Expected {} outs, given outputs_info of length {}, but `fn` returns {}. Pass None in outputs_info for returns which don't accumulate".format(len(outputs_info), len(outputs_info), len(inf_ret)))
    initializers = []
    for n in range(len(outputs_info)):
        if outputs_info[n] is not None:
            initializers.append(outputs_info[n])
        else:
            initializers.append(0. * inf_ret[n])
    def wrapwrap(nonepos, initializers):
        type_class = "list" if isinstance(initializers, list) else "tuple"
        def fnwrap(accs, inps):
            inps_then_accs = inps + [a for n, a in enumerate(accs) if n not in nonepos]
            fn_rets = fn(*inps_then_accs)
            return [fr for fr in fn_rets]
        return fnwrap
    this_fn = wrapwrap(nonepos, initializers)
    r = tf.scan(this_fn, sequences, initializers)
    return r
