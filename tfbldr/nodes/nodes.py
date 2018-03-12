import tensorflow as tf
import numpy as np
from scipy import linalg
from scipy.stats import truncnorm
from scipy.misc import factorial
import tensorflow as tf

from ..core import _get_name
from ..core import get_logger
from ..core import _get_name
from ..core import _get_shared
from ..core import _set_shared
from ..core import get_params_dict
from ..core import _shape
from ..core import _ndim
from ..core import dot
from ..core import scan
from ..core import get_weight_norm_default
from ..core import get_strict_mode_default

logger = get_logger()


def sigmoid(x):
    return tf.sigmoid(x)


def Sigmoid(x):
    return sigmoid(x)


def tanh(x):
    return tf.tanh(x)


def Tanh(x):
    return tanh(x)


def relu(x):
    return tf.nn.relu(x)


def ReLU(x):
    return relu(x)


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
              (lower - mu) / float(sigma), (upper - mu) / float(sigma),
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
    bound = scale * np.sqrt(3. / float(kern_sum))  # sqrt(3) for std of uniform
    return random_state.uniform(low=-bound, high=bound, size=shp).astype(
        "float32")


def np_glorot_uniform(shape, random_state, scale=1.):
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
        with 1. * sqrt(6 / (n_in + n_out)) scale
    Returns
    -------
    initialized_scaled, array-like
        Array-like of random values the same size as shape parameter
    """
    shp = shape
    kern_sum = sum(shp)
    bound = scale * np.sqrt(6. / float(kern_sum))
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
    You *must* get a list back, even for 1 element
    blah, = make_weights(...)
    or
    [blah] = make_weights(...)
    """
    ff = [None] * len(out_dims)
    fs = [scale] * len(out_dims)
    for i, out_dim in enumerate(out_dims):
        if init is None:
            if in_dim == out_dim:
                ff[i] = np_ortho
                fs[i] = 1.
            else:
                ff[i] = np_variance_scaled_uniform
                fs[i] = 1.
        elif init == "ortho":
            if in_dim != out_dim:
                raise ValueError("Unable to use ortho init for non-square matrices!")
            ff[i] = np_ortho
            fs[i] = 1.
        elif init == "glorot_uniform":
            ff[i] = np_glorot_uniform
        elif init == "normal":
            ff[i] = np_normal
            fs[i] = 0.01
        elif init == "truncated_normal":
            ff[i] = np_truncated_normal
            fs[i] = 0.075
        elif init == "embedding_normal":
            ff[i] = np_truncated_normal
            fs[i] = 1. / np.sqrt(in_dim)
        else:
            raise ValueError("Unknown init type %s" % init)

    ws = []
    for i, out_dim in enumerate(out_dims):
        if fs[i] == "default":
            wi = ff[i]((in_dim, out_dim), random_state)
            if len(wi.shape) == 4:
                wi = wi.transpose(2, 3, 1, 0)
            ws.append(wi)
        else:
            wi = ff[i]((in_dim, out_dim), random_state, scale=fs[i])
            if len(wi.shape) == 4:
                wi = wi.transpose(2, 3, 1, 0)
            ws.append(wi)
    return ws


def VqEmbedding(input_tensor, input_dim, output_dim,
                random_state=None, init="embedding_normal",
                scale="default",
                strict=None, name=None):
    if random_state is None:
        raise ValueError("Must pass instance of np.random.RandomState!")
    if init != "embedding_normal":
        raise ValueError("Other init values besides 'embedding_normal' not yet supported, got {}".format(init))
    if scale != "default":
        raise ValueError("Scale values besides 'default' not yet supported, got {}".format(scale))

    if name is None:
        name = _get_name()

    name_w = name + "_vqembedding_w"
    name_out = name + "_vqembedding_out"

    if strict is None:
        strict = get_strict_mode_default()

    if strict:
        cur_defs = get_params_dict()
        if name_w in cur_defs:
            raise ValueError("Name {} already created in params dict!".format(name_w))

    try:
        vectors = _get_shared(name_w)
    except NameError:
        logger.info("VqEmbedding layer {} initialized using init {}".format(name, init))
        embedding_weight, = make_numpy_weights(input_dim, [output_dim],
                                               random_state, init=init,
                                               scale=scale)
        embedding_weight = embedding_weight.transpose(1, 0)
        emb = tf.Variable(embedding_weight, trainable=True)
        _set_shared(name_w, emb)

    emb_r = tf.transpose(emb, (1, 0))
    ishp = _shape(input_tensor)
    extender = [None] * (len(ishp) - 1)
    sq_diff = tf.square(input_tensor[..., None] - emb_r.__getitem__(extender))
    sum_sq_diff = tf.reduce_sum(sq_diff, axis=-2)
    discrete_latent_idx = tf.argmin(sum_sq_diff, axis=-1)
    shp = _shape(discrete_latent_idx)
    flat_idx = tf.cast(tf.reshape(discrete_latent_idx, (-1,)), tf.int32)
    lu_vectors = tf.nn.embedding_lookup(emb, flat_idx)
    shp2 = _shape(lu_vectors)
    z_q_x = tf.reshape(lu_vectors, (-1, shp[1], shp[2], shp2[-1]))
    z_q_x = tf.identity(z_q_x, name=name_out)
    return z_q_x, discrete_latent_idx, emb


def Embedding(indices, n_symbols, output_dim, random_state=None,
              init="gaussian",
              strict=None, name=None):
    raise ValueError("Embedding not yet finalized - return embedded vectors and embed space itself")
    """
    Last dimension of indices tensor must be 1!!!!
    """
    if name is None:
        name = _get_name()

    if random_state is None:
        raise ValueError("Must pass random_state argument to Embedding")

    name_w = name + "_embedding_w"

    if strict is None:
        strict = get_strict_mode_default()

    if strict:
        cur_defs = get_params_dict()
        if name_w in cur_defs:
            raise ValueError("Name {} already created in params dict!".format(name_w))

    if init != "gaussian":
        raise ValueError("Currently unsupported init type {}".format(init))

    try:
        vectors = _get_shared(name_w)
    except NameError:
        logger.info("Linear layer {} initialized using init {}".format(name, init))
        vectors_weight = random_state.randn(n_symbols, output_dim).astype("float32")
        vectors = tf.Variable(vectors_weight, trainable=True)
        _set_shared(name_w, vectors)

    ii = tf.cast(indices, "int32")
    shp = _shape(ii)
    nd = _ndim(ii)
    if shp[-1] != 1:
        if nd < 3:
            logger.info("Embedding input should have last dimension 1, inferring dimension to 1, from shape {} to {}".format(shp, tuple(list(shp) + [1])))
            ii = tf.expand_dims(ii, axis=-1)
        else:
            raise ValueError("Embedding layer input must have last dimension 1 for input size > 3D, got {}".format(shp))

    shp = shape(ii)
    nd = len(shp)
    lu = tf.nn.embedding_lookup(vectors, ii)
    if nd == 3:
        lu = lu[:, :, 0]
    elif nd == 2:
        lu = lu[:, 0]
    else:
        raise ValueError("Input dimension not handled, Embedding input shape {} results in shape {}".format(shp, shape(lu)))
    return lu, vectors


def Linear(list_of_inputs, list_of_input_dims, output_dim, random_state=None,
           name=None, init=None, scale="default", biases=True, bias_offset=0.,
           strict=None):
    if random_state is None:
        raise ValueError("Must pass instance of np.random.RandomState!")
    nd = _ndim(list_of_inputs[0])
    input_var = tf.concat(list_of_inputs, axis=nd - 1)
    input_dim = sum(list_of_input_dims)
    if init is None or type(init) is str:
        logger.info("Linear layer {} initialized using init {}".format(name, init))
        weight_values, = make_numpy_weights(input_dim, [output_dim],
                                            random_state=random_state,
                                            init=init, scale=scale)
    else:
        # rely on announcement from parent class
        weight_values=init[0]

    if name is None:
        name = _get_name()

    name_w = name + "_linear_w"
    name_b = name + "_linear_b"
    name_out = name + "_linear_out"

    if strict is None:
        strict = get_strict_mode_default()

    if strict:
        cur_defs = get_params_dict()
        if name_w in cur_defs:
            raise ValueError("Name {} already created in params dict!".format(name_w))

        if name_b in cur_defs:
            raise ValueError("Name {} already created in params dict!".format(name_b))

    try:
        weight = _get_shared(name_w)
    except NameError:
        weight = tf.Variable(weight_values, trainable=True, name=name_w)
        _set_shared(name_w, weight)

    out = dot(input_var, weight)

    if biases:
        if (init is None) or (type(init) is str):
            b, = make_numpy_biases([output_dim])
        else:
            b = init[1]
        b = b + bias_offset
        try:
            biases = _get_shared(name_b)
        except NameError:
            biases = tf.Variable(b, trainable=True, name=name_b)
            _set_shared(name_b, biases)
        out = out + biases
    out = tf.identity(out, name=name_out)
    return out


def Conv2d(list_of_inputs, list_of_input_dims, num_feature_maps,
           kernel_size=(3, 3),
           dilation=[1, 1, 1, 1],
           strides=[1, 1, 1, 1],
           border_mode="same",
           init=None, scale="default",
           biases=True, bias_offset=0.,
           name=None, random_state=None, strict=None):
    if name is None:
        name = _get_name()

    if random_state is None:
        raise ValueError("Must pass instance of np.random.RandomState!")

    if strides != [1, 1, 1, 1]:
        try:
            int(strides)
            strides = [1, int(strides), int(strides), 1]
        except:
            raise ValueError("Changing strides by non-int not yet supported")

    if dilation != [1, 1, 1, 1]:
        raise ValueError("Changing dilation not yet supported")

    input_t = tf.concat(list_of_inputs, axis=-1)
    input_channels = sum(list_of_input_dims)
    input_height = _shape(input_t)[1]
    input_width = _shape(input_t)[2]

    name_w = name + "_conv2d_w"
    name_b = name + "_conv2d_b"
    name_out = name + "_conv2d_out"
    if strict is None:
        strict = get_strict_mode_default()

    if strict:
        cur_defs = get_params_dict()
        if name_w in cur_defs:
            raise ValueError("Name {} already created in params dict!".format(name_w))

        if name_b in cur_defs:
            raise ValueError("Name {} already created in params dict!".format(name_b))

    weight_values, = make_numpy_weights((input_channels, input_width, input_height),
                                        [(num_feature_maps, kernel_size[0], kernel_size[1])],
                                        init=init,
                                        scale=scale,
                                        random_state=random_state)
    try:
        weight = _get_shared(name_w)
    except NameError:
        logger.info("Conv2d layer {} initialized using init {}".format(name, init))
        weight = tf.Variable(weight_values, trainable=True, name=name_w)
        _set_shared(name_w, weight)

    if border_mode == "same":
        pad = "SAME"
    elif border_mode == "valid":
        pad = "VALID"
    else:
        try:
            int(border_mode)
            new_pad = [0, int(border_mode), int(border_mode), 0]
        except:
            try:
                # assume it is a custom list border pad
                # https://stackoverflow.com/questions/37659538/custom-padding-for-convolutions-in-tensorflow
                new_pad = [int(bi) for bi in border_mode]
            except:
                raise ValueError("Unknown border_mode {} specified".format(border_mode))

        assert len(new_pad) == 4
        input_t = tf.pad(input_t, [[new_pad[0]] * 2,
                                   [new_pad[1]] * 2,
                                   [new_pad[2]] * 2,
                                   [new_pad[3]] * 2], "CONSTANT")
        pad = "VALID"

    out = tf.nn.conv2d(input_t, weight, strides, padding=pad)
    if biases:
        if (init is None) or (type(init) is str):
            b, = make_numpy_biases([num_feature_maps])
        else:
            b = init[1]
        b = b + bias_offset
        try:
            biases = _get_shared(name_b)
        except NameError:
            biases = tf.Variable(b, trainable=True, name=name_b)
            _set_shared(name_b, biases)
        out = out + biases[None, None, None]
    out = tf.identity(out, name=name_out)
    return out


def ConvTranspose2d(list_of_inputs, list_of_input_dims, num_feature_maps,
                    kernel_size=(3, 3),
                    strides=None,
                    border_mode="same",
                    init=None, scale="default",
                    biases=True, bias_offset=0.,
                    name=None, random_state=None, strict=None):
    if name is None:
        name = _get_name()

    if random_state is None:
        raise ValueError("Must pass instance of np.random.RandomState!")

    if strides is None:
        raise ValueError("Conv2dTranspose is nearly always used with strides > 1!")

    if strides != [1, 1, 1, 1]:
        try:
            int(strides)
            strides = [1, int(strides), int(strides), 1]
        except:
            raise ValueError("Changing strides by non-int not yet supported")

    input_t = tf.concat(list_of_inputs, axis=-1)
    input_channels = sum(list_of_input_dims)
    input_height = _shape(input_t)[1]
    input_width = _shape(input_t)[2]

    name_w = name + "_convtranspose2d_w"
    name_b = name + "_convtranspose2d_b"
    name_out = name + "_convtranspose2d_out"
    if strict is None:
        strict = get_strict_mode_default()

    if strict:
        cur_defs = get_params_dict()
        if name_w in cur_defs:
            raise ValueError("Name {} already created in params dict!".format(name_w))

        if name_b in cur_defs:
            raise ValueError("Name {} already created in params dict!".format(name_b))

    weight_values, = make_numpy_weights((input_channels, input_width, input_height),
                                        [(num_feature_maps, kernel_size[0], kernel_size[1])],
                                        init=init,
                                        scale=scale,
                                        random_state=random_state)
    # transpose out and in to match transpose behavior
    # TODO: does this also change init scales?
    weight_values = weight_values.transpose(0, 1, 3, 2)
    try:
        weight = _get_shared(name_w)
    except NameError:
        logger.info("ConvTranspose2d layer {} initialized using init {}".format(name, init))
        weight = tf.Variable(weight_values, trainable=True, name=name_w)
        _set_shared(name_w, weight)

    if border_mode == "same":
        pad = "SAME"
    elif border_mode == "valid":
        pad = "VALID"
    else:
        try:
            int(border_mode)
            new_pad = [0, int(border_mode), int(border_mode), 0]
        except:
            try:
                # assume it is a custom list border pad
                # https://stackoverflow.com/questions/37659538/custom-padding-for-convolutions-in-tensorflow
                new_pad = [int(bi) for bi in border_mode]
            except:
                raise ValueError("Unknown border_mode {} specified".format(border_mode))

        assert len(new_pad) == 4
        """
        input_t = tf.pad(input_t, [[new_pad[0]] * 2,
                                   [new_pad[1]] * 2,
                                   [new_pad[2]] * 2,
                                   [new_pad[3]] * 2], "CONSTANT")
        """
        pad = "SAME"

    shp = _shape(input_t)
    btch_sz = shp[0]
    # calcs from PyTorch docs
    output_shape = tf.stack([btch_sz,
                            (input_height - 1) * strides[1] - 2 * new_pad[1] + kernel_size[0],
                            (input_width - 1) * strides[2] - 2 * new_pad[2] + kernel_size[1],
                            num_feature_maps])
    out = tf.nn.conv2d_transpose(input_t, weight, output_shape, strides, padding=pad)

    if biases:
        if (init is None) or (type(init) is str):
            b, = make_numpy_biases([num_feature_maps])
        else:
            b = init[1]
        b = b + bias_offset
        try:
            biases = _get_shared(name_b)
        except NameError:
            biases = tf.Variable(b, trainable=True, name=name_b)
            _set_shared(name_b, biases)
        out = out + biases[None, None, None]
    out = tf.identity(out, name=name_out)
    return out


def BatchNorm2d(input_tensor, train_test_flag,
                gamma_init=1., beta_init=0.,
                decay=0.9,
                eps=1E-3,
                affine=True,
                strict=None,
                name=None):
    # https://r2rt.com/implementing-batch-normalization-in-tensorflow.html
    if name is None:
        name = _get_name()

    name_scale = name + "_batchnorm_s"
    name_beta = name + "_batchnorm_b"
    name_out = name + "_batchnorm_out"
    if strict is None:
        strict = get_strict_mode_default()

    if strict:
        cur_defs = get_params_dict()
        if name_scale in cur_defs:
            raise ValueError("Name {} already created in params dict!".format(name_scale))

        if name_beta in cur_defs:
            raise ValueError("Name {} already created in params dict!".format(name_beta))

    try:
        scale = _get_shared(name_scale)
    except NameError:
        scale = tf.Variable(gamma_init * tf.ones([input_tensor.get_shape()[-1]]), trainable=True, name=name_scale)
        _set_shared(name_scale, scale)

    try:
        beta = _get_shared(name_beta)
    except NameError:
        beta = tf.Variable(beta_init * tf.ones([input_tensor.get_shape()[-1]]), trainable=True, name=name_beta)
        _set_shared(name_beta, beta)

    pop_mean = tf.Variable(tf.zeros([input_tensor.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([input_tensor.get_shape()[-1]]), trainable=False)

    shp = _shape(input_tensor)
    def left():
        batch_mean, batch_var = tf.nn.moments(input_tensor, list(range(len(shp)))[:-1])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(input_tensor,
                batch_mean, batch_var, beta, scale, eps)

    def right():
        return tf.nn.batch_normalization(input_tensor,
            pop_mean, pop_var, beta, scale, eps)

    out = tf.cond(train_test_flag <= 0.5, lambda: left(), lambda: right())
    return tf.identity(out, name=name_out)


def SimpleRNNCell(list_of_inputs, list_of_input_dims, previous_hidden,
                  num_units, output_dim, random_state=None,
                  name=None, init=None, scale="default", strict=None):
    # output is the thing to use in following layers, state is a tuple that contains things to feed into the next call
    if random_state is None:
        raise ValueError("Must pass random_state")

    if name is None:
        name = _get_name()
    hidden_dim = num_units
    inp_to_h = Linear(list_of_inputs, list_of_input_dims, hidden_dim, random_state=random_state,
                      name=name + "_simple_rnn_inp_to_h",
                      init=init, strict=strict)
    h_to_h = Linear([previous_hidden], [hidden_dim], hidden_dim, random_state=random_state,
                    name=name + "_simple_rnn_h_to_h", biases=False,
                    init=init, strict=strict)
    h = tf.nn.tanh(inp_to_h + h_to_h)
    h_to_out = Linear([h], [hidden_dim], output_dim, random_state=random_state,
                      name=name + "_simple_rnn_h_to_out",
                      init=init, strict=strict)
    return h_to_out, (h,)


def _dropout(tensor, keep_threshold, seed):
    sample = tf.where(tf.random_uniform([], minval=0., maxval=1., seed=seed) < keep_threshold,
                      tf.ones([]), tf.zeros([]))
    return sample * tensor


def LSTMCell(list_of_inputs, list_of_input_dims,
             previous_hidden, previous_cell,
             num_units,
             output_dim=None,
             input_mask=None,
             random_state=None,
             name=None, init=None, scale="default",
             forget_bias=1.,
             cell_dropout=None,
             strict=None):
    # cell_dropout should be a value in [0., 1.], or None
    # output is the thing to use in following layers, state is a tuple that feeds into the next call
    if random_state is None:
        raise ValueError("Must pass random_state")

    if name is None:
        name = _get_name()

    input_dim = sum(list_of_input_dims)
    hidden_dim = 4 * num_units

    if init is None or init == "truncated_normal":
        inp_init = "truncated_normal"
        h_init = "truncated_normal"
        out_init = "truncated_normal"
    elif init == "glorot_uniform":
        inp_init = "glorot_uniform"
        h_init = "glorot_uniform"
        out_init = "glorot_uniform"
    elif init == "normal":
        inp_init = "normal"
        h_init = "normal"
        out_init = "normal"
    else:
        raise ValueError("Unknown init argument {}".format(init))

    comb_w_np, = make_numpy_weights(input_dim + num_units, [hidden_dim],
                                    random_state=random_state,
                                    init=inp_init)
    comb_b_np, = make_numpy_biases([hidden_dim])

    logger.info("LSTMCell {} input to hidden initialized using init {}".format(name, inp_init))
    logger.info("LSTMCell {} hidden to hidden initialized using init {}".format(name, h_init))
    lstm_proj = Linear(list_of_inputs + [previous_hidden], list_of_input_dims + [hidden_dim],
                       hidden_dim,
                       random_state=random_state,
                       name=name + "_lstm_proj",
                       init=(comb_w_np, comb_b_np), strict=strict)

    i, j, f, o = tf.split(lstm_proj, 4, axis=-1)

    if cell_dropout is not None:
        pj = tf.nn.dropout(tf.tanh(j), cell_dropout,
                           seed=random_state.randint(0, 1E6))
    else:
        pj = tf.tanh(j)

    c = tf.sigmoid(f + forget_bias) * previous_cell + tf.sigmoid(i) * pj
    if input_mask is not None:
        c = input_mask[:, None] * c + (1. - input_mask[:, None]) * previous_cell

    h = tf.sigmoid(o) * tf.tanh(c)
    if input_mask is not None:
        h = input_mask[:, None] * h + (1. - input_mask[:, None]) * h

    if output_dim is not None:
        h_to_out_w_np, = make_numpy_weights(num_units, [output_dim],
                                            random_state=random_state,
                                            init=out_init)
        h_to_out_b_np, = make_numpy_biases([output_dim])
        h_to_out = Linear([h], [num_units], output_dim, random_state=random_state,
                          name=name + "_lstm_h_to_out",
                          init=(h_to_out_w_np, h_to_out_b_np), strict=strict)
        final_out = h_to_out
        logger.info("LSTMCell {} hidden to output initialized using init {}".format(name, out_init))
    else:
        final_out = h
    return final_out, (h, c)


def GaussianAttentionCell(list_of_step_inputs, list_of_step_input_dims,
                          previous_state_list,
                          previous_attention_position,
                          full_conditioning_tensor,
                          full_conditioning_tensor_dim,
                          num_units,
                          previous_attention_weight,
                          att_dim=10,
                          attention_scale=1.,
                          cell_type="lstm",
                          name=None,
                          input_mask=None,
                          conditioning_mask=None,
                          random_state=None,
                          cell_dropout=None,
                          strict=None, init=None):
    #returns w_t, k_t, phi_t, state
    # where state is the state tuple retruned by the inner cell_type

    if name is None:
        name = _get_name()
    name = name + "_gaussian_attention"

    check = any([len(_shape(si)) != 2 for si in list_of_step_inputs])
    if check:
        raise ValueError("Unable to support step_input with n_dims != 2")

    if init is None or init == "truncated_normal":
        rnn_init = "truncated_normal"
        forward_init = "truncated_normal"
    else:
        raise ValueError("init != None not supported")

    if cell_type == "gru":
        raise ValueError("NYI")
    elif cell_type == "lstm":
        att_rnn_out, state = LSTMCell(list_of_step_inputs + [previous_attention_weight],
                                      list_of_step_input_dims + [full_conditioning_tensor_dim],
                                      previous_state_list[0], previous_state_list[1],
                                      num_units,
                                      input_mask=input_mask,
                                      random_state=random_state,
                                      cell_dropout=cell_dropout,
                                      name=name + "_gauss_att_lstm",
                                      init=rnn_init)
    else:
        raise ValueError("Unsupported cell_type %s" % cell_type)

    ret = Linear(
        list_of_inputs=[att_rnn_out], list_of_input_dims=[num_units],
        output_dim=3 * att_dim, name=name + "_group",
        random_state=random_state,
        strict=strict, init=forward_init)
    a_t = ret[:, :att_dim]
    b_t = ret[:, att_dim:2 * att_dim]
    k_t = ret[:, 2 * att_dim:]

    k_tm1 = previous_attention_position
    cond_dim = full_conditioning_tensor_dim
    ctx = full_conditioning_tensor
    ctx_mask = conditioning_mask

    """
    ctx = Linear(
        list_of_inputs=[full_conditioning_tensor],
        list_of_input_dims=[full_conditioning_tensor_dim],
        output_dim=next_proj_dim, name=name + "_proj_ctx",
        weight_norm=weight_norm,
        random_state=random_state,
        strict=strict, init=ctx_forward_init)
    """

    a_t = tf.exp(a_t)
    b_t = tf.exp(b_t)
    a_t = tf.identity(a_t, name=name + "_a_scale")
    b_t = tf.identity(b_t, name=name + "_b_scale")
    step_size = attention_scale * tf.exp(k_t)
    k_t = k_tm1 + step_size
    k_t = tf.identity(k_t, name=name + "_position")

    # tf.shape and tensor.shape are not the same...
    u = tf.cast(tf.range(0., limit=tf.shape(full_conditioning_tensor)[0], delta=1.), dtype=tf.float32)
    u = tf.expand_dims(tf.expand_dims(u, axis=0), axis=0)

    def calc_phi(lk_t, la_t, lb_t, lu):
        la_t = tf.expand_dims(la_t, axis=2)
        lb_t = tf.expand_dims(lb_t, axis=2)
        lk_t = tf.expand_dims(lk_t, axis=2)
        phi = tf.exp(-tf.square(lk_t - lu) * lb_t) * la_t
        phi = tf.reduce_sum(phi, axis=1, keep_dims=True)
        return phi

    phi_t = calc_phi(k_t, a_t, b_t, u)
    phi_t = tf.identity(phi_t, name=name + "_phi")

    """
        # Notes from pytorch tests
        # sanity check shapes for proper equivalent to np.dot
        aaaa = np.random.randn(50, 1, 46)
        bbbb = np.random.randn(50, 46, 400)
        r = np.matmul(aaaa, bbbb)
        # r has shape ms, 1, embed_dim
        # since aaaa and bbbb are > 2d, treated as stack of matrices, matrix dims on last 2 axes
        # this means 50, 1, 46 x 50, 46, 400 is 50 reps of 1, 46 x 46, 400
        # leaving shape 50, 1, 400
        # equivalent to dot for 1 matrix is is (aaaa[0][:, :, None] * bbbb[0][None, :, :]).sum(axis=-2)
        # so for all 50, (aaaa[:, :, :, None] * bbbb[:, None, :, :]).sum(axis=-2)
        # ((aaaa[:, :, :, None] * bbbb[:, None, :, :]).sum(axis=-2) == r).all()
        _a = Variable(th.FloatTensor(aaaa))
        _b = Variable(th.FloatTensor(bbbb))
        e_a = _a[:, :, :, None].expand(_a.size(0), _a.size(1), _a.size(2), _b.size(2))
        e_b = _b[:, None, :, :].expand(_b.size(0), _a.size(1), _b.size(1), _b.size(2))
        # In [17]: np.sum(((e_a * e_b).sum(dim=-2)[:, :, 0].data.numpy() - r) ** 2)
        # Out[17]: 1.6481219193765024e-08
        # equivalent to comb = th.matmul(phi, c), for backwards compat
        e_phi = phi[:, :, :, None].expand(phi.size(0), phi.size(1), phi.size(2), c.size(2))
        e_c = c[:, None, :, :].expand(c.size(0), phi.size(1), c.size(1), c.size(2))
        comb = (e_phi * e_c).sum(dim=-2)[:, :, 0]
        # comb has shape minibatch_size, 1, embed_size
        # w_t has shape minibatch_size, embed_size
        w_t = comb[:, 0, :]
    """
    if conditioning_mask is not None:
        w_t_pre = phi_t * tf.transpose(ctx, (1, 2, 0))
        w_t_masked = w_t_pre * (tf.transpose(ctx_mask, (1, 0))[:, None])
        w_t = tf.reduce_sum(w_t_masked, axis=-1)[:, None]
    else:
        w_t = tf.matmul(phi_t, tf.transpose(ctx, (1, 0, 2)))
    phi_t = phi_t[:, 0]
    w_t = w_t[:, 0]
    w_t = tf.identity(w_t, name=name + "_post_weighting")
    return w_t, k_t, phi_t, state


def BernoulliAndCorrelatedGMM(
        list_of_inputs, list_of_input_dims, bias=None, output_dim=2, n_components=10,
        name=None, random_state=None, strict=None, init=None):
    """
    returns bernoulli, coeffs, mus, sigmas, corr
    """
    assert n_components >= 1
    if name is None:
        name = _get_name()
    else:
        name = name + "_bmdn"

    if output_dim != 2:
        raise ValueError("General calculation for GMM not yet implemented")

    e = Linear(list_of_inputs, list_of_input_dims, 1,
               random_state=random_state,
               init=init, name=name + "_bern")
    pi = Linear(list_of_inputs, list_of_input_dims, n_components,
                random_state=random_state,
                init=init, name=name + "_coeff")
    mu1 = Linear(list_of_inputs, list_of_input_dims, n_components,
                 random_state=random_state,
                 init=init, name=name + "_mu1")
    mu2 = Linear(list_of_inputs, list_of_input_dims, n_components,
                 random_state=random_state,
                 init=init, name=name + "_mu2")
    std1 = Linear(list_of_inputs, list_of_input_dims, n_components,
                  random_state=random_state,
                  init=init, name=name + "_std1")
    std2 = Linear(list_of_inputs, list_of_input_dims, n_components,
                  random_state=random_state,
                  init=init, name=name + "_std2")
    rho = Linear(list_of_inputs, list_of_input_dims, n_components,
                 random_state=random_state,
                 init=init, name=name + "_corr")

    if bias is None:
        bias = 0.

    return tf.nn.sigmoid(e), \
           tf.nn.softmax(pi * (1. + bias), dim=-1), \
           mu1, mu2, \
           tf.exp(std1 - bias), tf.exp(std2 - bias), \
           tf.nn.tanh(rho)


def BernoulliAndCorrelatedGMMCost(
    bernoulli_values, coeff_values, mu_values_list, sigma_values_list,
    corr_values, true_values_bernoulli, true_values_coord_list, name=None):
    """
    Bernoulli combined with correlated gaussian mixture model negative log
    likelihood compared to true_values.

    This is typically paired with BernoulliAndLogitGMM

    Based on implementation from Junyoung Chung.

    Parameters
    ----------
    bernoulli_values : tensor, shape
        The predicted values out of some layer, normally a sigmoid layer
    coeff_values : tensor, shape
        The predicted values out of some layer, normally a softmax layer
    mu_values_list: tensor, shape
        The predicted values out of some layer, normally a linear layer
    sigma_values_list: tensor, shape
        list of predicted values out of some layer, normally an exp or softplus layer
    corr_values: tensor, shape
    true_values_bernoulli : tensor, shape[:-1]
        Ground truth values. Must be the same shape as mu_values.shape[:-1],
        assumes the bernoulli true values are on the first entry ([:, :, 0])
    true_values_coords_list :
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

    xs = true_values_coord_list[0]
    ys = true_values_coord_list[1]
    es = true_values_bernoulli
    txs = _shape(xs)
    if txs[-1] != 1:
        raise ValueError("Targets must be 1 dimensional")
    tys = _shape(ys)
    tes = _shape(es)
    if tys != txs:
        raise ValueError("Targets must have the same dimension")
    if tes != txs:
        raise ValueError("Targets must have the same dimension")

    # seq length generally -1
    batch_size = txs[1]
    def _2d(a):
        return tf.reshape(a, (-1, _shape(a)[-1]))

    true_values_bernoulli = _2d(true_values_bernoulli)
    true_values_coord_list = [_2d(tvc) for tvc in true_values_coord_list]
    coeff_values = _2d(coeff_values)
    bernoulli_values = _2d(bernoulli_values)
    corr_values = _2d(corr_values)
    mu_values_list = [_2d(mv) for mv in mu_values_list]
    sigma_values_list = [_2d(sv) for sv in sigma_values_list]

    error_msg = "Dimension of variable {} not supported, got {}. Must be 2"
    if len(_shape(true_values_bernoulli)) != 2:
        raise ValueError(error_msg.format("true_values_bernoulli", len(_shape(true_values_bernoulli))))
    elif any([len(_shape(tvc)) != 2 for tvc in true_values_coord_list]):
        raise ValueError(error_msg.format("true_values_coord_list", [len(_shape(true_values_coord_list[0])), len(_shape(truce_values_coord_list[1]))]))
    elif len(_shape(bernoulli_values)) != 2:
        raise ValueError(error_msg.format("bernoulli_values", len(_shape(bernoulli_values))))
    elif len(_shape(coeff_values)) != 2:
        raise ValueError(error_msg.format("coeff_values", len(_shape(coeff_values))))
    elif any([len(_shape(m)) != 2 for m in mu_values_list]):
        raise ValueError(error_msg.format("mu_values", [len(_shape(mu_values[0])), len(_shape(mu_values_list[1]))]))
    elif any([len(_shape(s)) != 2 for s in sigma_values_list]):
        raise ValueError(error_msg.format("sigma_values", [len(_shape(sigma_values[0])), len(_shape(sigma_values[1]))]))
    elif len(_shape(corr_values)) != 2:
        raise ValueError(error_msg.format("corr_values", len(_shape(corr_values))))


    if len(true_values_coord_list) != 2:
        raise ValueError("Only 2D GMM currently supported, got {} inputs in list for true coordinates".format(len(true_values_coord_list)))

    if len(mu_values_list) != 2:
        raise ValueError("Only 2D GMM currently supported, got {} inputs in list for mu values".format(len(true_values_coord_list)))

    if len(sigma_values_list) != 2:
        raise ValueError("Only 2D GMM currently supported, got {} inputs in list for sigma values".format(len(true_values_coord_list)))

    mu_1 = mu_values_list[0]
    mu_1 = tf.identity(mu_1, name=name + "_mu_1")
    mu_2 = mu_values_list[1]
    mu_2 = tf.identity(mu_2, name=name + "_mu_2")

    corr_values = tf.identity(corr_values, name=name + "_corrs")

    sigma_1 = sigma_values_list[0]
    sigma_1 = tf.identity(sigma_1, name=name + "_sigma_1")
    sigma_2 = sigma_values_list[1]
    sigma_2 = tf.identity(sigma_2, name=name + "_sigma_2")

    bernoulli_values = tf.identity(bernoulli_values, name=name + "_bernoullis")
    coeff_values = tf.identity(coeff_values, name=name + "_coeffs")

    true_0 = true_values_bernoulli
    true_1 = true_values_coord_list[0]
    true_2 = true_values_coord_list[1]

    # don't be clever
    buff = (1. - tf.square(corr_values)) + 1E-6
    x_term = (true_1 - mu_1) / sigma_1
    y_term = (true_2 - mu_2) / sigma_2

    Z = tf.square(x_term) + tf.square(y_term) - 2. * corr_values * x_term * y_term
    N = 1. / (2. * np.pi * sigma_1 * sigma_2 * tf.sqrt(buff)) * tf.exp(-Z / (2. * buff))
    ep = true_0 * bernoulli_values + (1. - true_0) * (1. - bernoulli_values)
    assert _shape(ep)[-1] == 1
    ep = ep[:, 0]
    rp = tf.reduce_sum(coeff_values * N, axis=-1)
    nll = -tf.log(rp + 1E-8) - tf.log(ep + 1E-8)
    nll = tf.reshape(nll, (-1, batch_size))
    return nll


def BernoulliCrossEntropyCost(predicted, target, eps=1E-8):
    shpp = _shape(predicted)
    shpt = _shape(target)
    for i in range(len(shpp)):
        if shpt[i] != -1 and shpt[i] != shpp[i]:
            raise ValueError("Shape mismatch between predicted {} and target {}".format(shpp, shpt))
    if shpt[-1] != 1 and shpp[-1] != 1:
        raise ValueError("Shape last dimension must be 1, got predicted {} and target {}".format(shpp, shpt))
    ep = target * tf.log(predicted + eps) + (1. - target) * tf.log(1. - predicted + eps)
    return ep
