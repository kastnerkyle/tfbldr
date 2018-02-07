from __future__ import print_function
from extras import fetch_iamondb, rsync_fetch, list_iterator
import tensorflow as tf
import numpy as np
import argparse
import os

# based on
# https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
def sigmoid(x):
    gtz = x > 0
    gtz_z = np.exp(gtz * -x)
    gtz_z = gtz * (1. / (1 + gtz_z))

    ltz = x <= 0
    ltz_z = np.exp(ltz * x)
    ltz_z = ltz * (ltz_z / (1. + ltz_z))
    return (gtz_z + ltz_z)


def softmax(X, axis=-1):
    # https://nolanbconaway.github.io/blog/2017/softmax-numpy
    # make X at least 2d
    y = np.atleast_2d(X)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    # flatten if X was 1D
    if len(X.shape) == 1:
        p = p.flatten()
    return p


def split_strokes(points):
    # assumes points is pen, dx, dy
    points = np.array(points)
    strokes = []
    b = 0
    for e in range(len(points)):
        if points[e, 0] == 1.:
            strokes += [points[b: e + 1, 1:].copy()]
            b = e + 1
        elif e == (len(points) - 1):
            # get last stroke
            strokes += [points[b: e + 1, 1:].copy()]
    return strokes


def sample(mu1, mu2, std1, std2, rho, pi, bernoulli):
    # todo, vectorize...
    gs = []
    for i in range(len(pi)):
        g = np.random.choice(np.arange(pi[i].shape[0]), p=pi[i])
        gs.append(g)
    sel = np.array(gs)
    mu1 = mu1[np.arange(len(mu1)), sel]
    mu2 = mu2[np.arange(len(mu2)), sel]
    std1 = std1[np.arange(len(std1)), sel]
    std2 = std2[np.arange(len(std2)), sel]
    rho = rho[np.arange(len(rho)), sel]
    covs = np.array([[std1 * std1, std1 * std2 * rho],
                    [std1 * std2 * rho, std2 * std2]])
    means = np.array([mu1, mu2])
    means = means.transpose(1, 0)
    covs = covs.transpose(2, 0, 1)
    points = []
    for ii in range(len(mu1)):
        x, y = np.random.multivariate_normal(means[ii], covs[ii])
        points.append(np.array([x, y]))
    points = np.array(points)
    pens = np.random.binomial(1, bernoulli)
    comb = np.concatenate([pens, points], axis=1)
    return comb


def numpy_sample_bernoulli_and_bivariate_gmm(mu, sigma, corr, coeff, binary,
                                             random_state, binary_threshold=0.5, epsilon=1E-5,
                                             use_map=False):
    # todo, vectorize
    # only handles one example at a time
    gs = []
    for i in range(len(coeff)):
        g = random_state.choice(np.arange(coeff[i].shape[0]), p=coeff[i])
        gs.append(g)
    idx = np.array(gs)

    mu_i = mu[np.arange(len(mu)), :, idx]
    sigma_i = sigma[np.arange(len(sigma)), :, idx]
    corr_i = corr[np.arange(len(corr)), idx]

    mu_x = mu_i[:, 0]
    mu_y = mu_i[:, 1]
    sigma_x = sigma_i[:, 0]
    sigma_y = sigma_i[:, 1]
    if use_map:
        s_b = binary > binary_threshold
        s_x = mu_x[:, None]
        s_y = mu_y[:, None]
        s = np.concatenate([s_b, s_x, s_y], axis=-1)
        return s
    else:
        z = random_state.randn(*mu_i.shape)
        un = random_state.rand(*binary.shape)
        s_b = un < binary

        s_x = (mu_x + sigma_x * z[:, 0])[:, None]
        s_y = mu_y + sigma_y * (
            (z[:, 0] * corr_i) + (z[:, 1] * np.sqrt(1. - corr_i ** 2)))
        s_y = s_y[:, None]
        s = np.concatenate([s_b, s_x, s_y], axis=-1)
        return s

parser = argparse.ArgumentParser()
parser.add_argument('--model', dest='model_path', type=str, default=os.path.join('pretrained', 'model-29'))
parser.add_argument('--text', dest='text', type=str, default=None)
parser.add_argument('--bias', dest='bias', type=float, default=1.)
parser.add_argument('--seed', dest='seed', type=int, default=1999)
parser.add_argument('--force', dest='force', action='store_true', default=False)
parser.add_argument('--noinfo', dest='info', action='store_false', default=True)
args = parser.parse_args()

iamondb = rsync_fetch(fetch_iamondb, "leto01")
data = iamondb["data"]
target = iamondb["target"]

X = [xx.astype("float32") for xx in data]
y = [yy.astype("float32") for yy in target]

train_itr = list_iterator([X, y], minibatch_size=50, axis=1,
                          stop_index=10000,
                          make_mask=True)
trace_mb, trace_mask, text_mb, text_mask = next(train_itr)
train_itr.reset()

n_letters = len(iamondb["vocabulary"])
random_state = np.random.RandomState(args.seed)
n_attention = 10
cut_len = 300
h_dim = 400
n_batch = trace_mb.shape[1]

# real sample
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(args.model_path + '.meta')
    saver.restore(sess, args.model_path)
    graph = tf.get_default_graph()
    init_h1_np = np.zeros((n_batch, h_dim)).astype("float32")
    init_h2_np = np.zeros((n_batch, h_dim)).astype("float32")
    init_h3_np = np.zeros((n_batch, h_dim)).astype("float32")
    init_att_h_np = np.zeros((n_batch, h_dim)).astype("float32")
    init_att_k_np = np.zeros((n_batch, n_attention)).astype("float32")
    init_att_w_np = np.zeros((n_batch, n_letters)).astype("float32")

    everything = [n.name for n in tf.get_default_graph().as_graph_def().node]

    X_char = graph.get_tensor_by_name("X_char:0")
    X_char_mask = graph.get_tensor_by_name("X_char_mask:0")
    y_pen = graph.get_tensor_by_name("y_pen:0")
    y_pen_mask = graph.get_tensor_by_name("y_pen_mask:0")
    init_h1 = graph.get_tensor_by_name("init_h1:0")
    init_h2 = graph.get_tensor_by_name("init_h2:0")
    init_att_h = graph.get_tensor_by_name("init_att_h:0")
    init_att_k = graph.get_tensor_by_name("init_att_k:0")
    init_att_w = graph.get_tensor_by_name("init_att_w:0")

    logit_bernoullis = graph.get_tensor_by_name("b_gmm_logit_bernoulli_and_correlated_logit_gaussian_mixture_logit_bernoullis:0")
    logit_coeffs = graph.get_tensor_by_name("b_gmm_logit_bernoulli_and_correlated_logit_gaussian_mixture_logit_coeffs:0")
    mus = graph.get_tensor_by_name("b_gmm_logit_bernoulli_and_correlated_logit_gaussian_mixture_mus:0")
    logit_sigmas = graph.get_tensor_by_name("b_gmm_logit_bernoulli_and_correlated_logit_gaussian_mixture_logit_sigmas:0")
    corrs = graph.get_tensor_by_name("b_gmm_logit_bernoulli_and_correlated_logit_gaussian_mixture_corrs:0")

    att_k = graph.get_tensor_by_name("att_k:0")
    att_w = graph.get_tensor_by_name("att_w:0")
    att_h = graph.get_tensor_by_name("att_h:0")
    h1_o = graph.get_tensor_by_name("h1_o:0")
    h2_o = graph.get_tensor_by_name("h2_o:0")

    all_res = [tmb for tmb in trace_mb[:2]]
    all_res_mask = [0. * tm + 1. for tm in trace_mask[:2]]
    all_att_h = [init_att_h_np,]
    all_att_k = [init_att_k_np,]
    all_att_w = [init_att_w_np,]
    all_h1 = [init_h1_np,]
    all_h2 = [init_h2_np,]
    sample_len = 2000
    monitor = 20
    for i in range(sample_len):
        if i == (sample_len - 1):
            print("Sampling step {}".format(i))
        elif i % monitor == 0:
            print("Sampling step {}".format(i))
        #this_trace_mb = np.array(all_res[-prime:])
        #this_trace_mask = np.array(all_res_mask[-prime:])
        this_trace_mb = np.array(all_res[-2:])
        this_trace_mask = np.array(all_res_mask[-2:])
        this_init_att_h = np.array(all_att_h[-1])
        this_init_att_k = np.array(all_att_k[-1])
        this_init_att_w = np.array(all_att_w[-1])
        this_init_h1 = np.array(all_h1[-1])
        this_init_h2 = np.array(all_h2[-1])

        feed = {X_char: text_mb,
                X_char_mask: text_mask,
                y_pen: this_trace_mb,
                y_pen_mask: this_trace_mask,
                init_h1: init_h1_np,
                init_h2: init_h2_np,
                init_att_h: init_att_h_np,
                init_att_k: init_att_k_np,
                init_att_w: init_att_w_np}

        desired_outs = [logit_bernoullis, logit_coeffs, mus, logit_sigmas, corrs,
                        att_h, att_k, att_w,
                        h1_o, h2_o]
        r_outs = sess.run(desired_outs, feed)

        logit_bernoullis_np = r_outs[0]
        logit_coeffs_np = r_outs[1]
        mus_np = r_outs[2]
        logit_sigmas_np = r_outs[3]
        corrs_np = r_outs[4]
        att_h_np = r_outs[5]
        att_k_np = r_outs[6]
        att_w_np = r_outs[7]
        h1_np = r_outs[8]
        h2_np = r_outs[9]

        bias = 1.
        sigmas_np = np.exp(logit_sigmas_np - bias)
        coeffs_np = softmax(logit_coeffs_np)
        #coeffs_np = softmax(logit_coeffs_np * (1. + bias))
        bernoullis_np = sigmoid(logit_bernoullis_np)

        this_res = []
        for choose in range(h1_np.shape[1]):
            mus_i = mus_np[:, choose]
            sigmas_i = sigmas_np[:, choose]
            corrs_i = corrs_np[:, choose]
            corrs_i = corrs_i[:, 0]
            bernoullis_i = bernoullis_np[:, choose]
            coeffs_i = coeffs_np[:, choose]
            coeffs_i = coeffs_i[:, 0]
            att_k_i = att_k_np[:, choose]
            att_w_i = att_w_np[:, choose]
            #res1 = sample(mus_i[:, 0], mus_i[:, 1], sigmas_i[:, 0], sigmas_i[:, 1], corrs_i, coeffs_i, bernoullis_i)
            res = numpy_sample_bernoulli_and_bivariate_gmm(mus_i, sigmas_i, corrs_i, coeffs_i, bernoullis_i,
                random_state=random_state)
            this_res.append(res)
        # now 1, minibatch_size, 3
        this_res = np.array(this_res).transpose(1, 0, 2)
        this_mask = 0. * this_res[-1, :, 0] + 1.
        all_res.append(this_res[0])
        all_res_mask.append(this_mask)
        all_att_h.append(att_h_np[0])# = [init_att_h_np,]
        all_att_k.append(att_k_np[0])# = [init_att_h_np,]
        all_att_w.append(att_w_np[0])# = [init_att_k_np,]
        all_h1.append(h1_np[0])
        all_h2.append(h2_np[0])

        #all_att_w = [init_att_w_np,]
        #all_h1 = [init_h1_np,]
        #all_h2 = [init_h2_np,]
        #all_h3 = [init_h3_np,]

        init_h1_np = h1_np[-1]
        init_h2_np = h2_np[-1]
        init_att_h_np = att_h_np[-1]
        init_att_k_np = att_k_np[-1]
        init_att_w_np = att_w_np[-1]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.exists("plots"):
        os.mkdir("plots")

    sampled_res = np.array(all_res)
    final_att_k = np.array(all_att_k)
    final_att_w = np.array(all_att_w)

    for choose in range(sampled_res.shape[1]):
        f, axarr = plt.subplots(2, 1)

        strokes = sampled_res[:, choose]
        final_att_k_i = final_att_k[:, choose]
        final_att_w_i = final_att_w[:, choose]

        strokes[:, 1:] = np.cumsum(strokes[:, 1:], axis=0)
        minx, maxx = np.min(strokes[:, 1]), np.max(strokes[:, 1])
        miny, maxy = np.min(strokes[:, 2]), np.max(strokes[:, 2])
        split = split_strokes(strokes)
        for sp in split:
            axarr[0].plot(sp[:, 0], sp[:, 1])
        axarr[1].imshow(final_att_w_i.T)
        #axarr[1].plot(final_att_k_i.mean(axis=-1))
        plt.savefig("plots/plot_results_{}.png".format(choose))
        plt.clf()
        plt.close()
    from IPython import embed; embed(); raise ValueError()

"""
# cheat sample
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(args.model_path + '.meta')
    saver.restore(sess, args.model_path)
    graph = tf.get_default_graph()
    init_h1_np = np.zeros((n_batch, h_dim)).astype("float32")
    init_h2_np = np.zeros((n_batch, h_dim)).astype("float32")
    init_h3_np = np.zeros((n_batch, h_dim)).astype("float32")
    init_att_h_np = np.zeros((n_batch, h_dim)).astype("float32")
    init_att_k_np = np.zeros((n_batch, n_attention)).astype("float32")
    init_att_w_np = np.zeros((n_batch, n_letters)).astype("float32")

    everything = [n.name for n in tf.get_default_graph().as_graph_def().node]

    X_char = graph.get_tensor_by_name("X_char:0")
    X_char_mask = graph.get_tensor_by_name("X_char_mask:0")
    y_pen = graph.get_tensor_by_name("y_pen:0")
    y_pen_mask = graph.get_tensor_by_name("y_pen_mask:0")
    init_h1 = graph.get_tensor_by_name("init_h1:0")
    init_h2 = graph.get_tensor_by_name("init_h2:0")
    init_h3 = graph.get_tensor_by_name("init_h3:0")
    init_att_h = graph.get_tensor_by_name("init_att_h:0")
    init_att_k = graph.get_tensor_by_name("init_att_k:0")
    init_att_w = graph.get_tensor_by_name("init_att_w:0")
    bias = graph.get_tensor_by_name("bias:0")
    log_bernoullis = graph.get_tensor_by_name("b_gmm_log_bernoulli_and_correlated_log_gaussian_mixture_log_bernoullis:0")
    coeffs = graph.get_tensor_by_name("b_gmm_log_bernoulli_and_correlated_log_gaussian_mixture_coeffs:0")
    mus = graph.get_tensor_by_name("b_gmm_log_bernoulli_and_correlated_log_gaussian_mixture_mus:0")
    log_sigmas = graph.get_tensor_by_name("b_gmm_log_bernoulli_and_correlated_log_gaussian_mixture_log_sigmas:0")
    corrs = graph.get_tensor_by_name("b_gmm_log_bernoulli_and_correlated_log_gaussian_mixture_corrs:0")

    att_k = graph.get_tensor_by_name("att_k:0")
    att_w = graph.get_tensor_by_name("att_w:0")

    feed = {X_char: text_mb,
            X_char_mask: text_mask,
            y_pen: trace_mb,
            y_pen_mask: trace_mask,
            init_h1: init_h1_np,
            init_h2: init_h2_np,
            init_h3: init_h3_np,
            init_att_h: init_att_h_np,
            init_att_k: init_att_k_np,
            init_att_w: init_att_w_np}

    desired_outs = [log_bernoullis, coeffs, mus, log_sigmas, corrs]
    r_outs = sess.run(desired_outs, feed)

    desired_atts = [att_k, att_w]
    r_atts = sess.run(desired_atts, feed)
    att_k_np = r_atts[0]
    att_w_np = r_atts[1]

    log_bernoullis_np = r_outs[0]
    coeffs_np = r_outs[1]
    mus_np = r_outs[2]
    log_sigmas_np = r_outs[3]
    corrs_np = r_outs[4]
    bernoullis_np = sigmoid(log_bernoullis_np)
    bias = 100.
    sigmas_np = np.exp(log_sigmas_np - bias)

    choose = 45
    mus_i = mus_np[:, choose]
    sigmas_i = sigmas_np[:, choose]
    corrs_i = corrs_np[:, choose]
    corrs_i = corrs_i[:, 0]
    bernoullis_i = bernoullis_np[:, choose]
    coeffs_i = coeffs_np[:, choose]
    coeffs_i = coeffs_i[:, 0]
    att_k_i = att_k_np[:, choose]
    att_w_i = att_w_np[:, choose]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f, axarr = plt.subplots(2, 1)

    res1 = sample(mus_i[:, 0], mus_i[:, 1], sigmas_i[:, 0], sigmas_i[:, 1], corrs_i, coeffs_i, bernoullis_i)
    res1 = res1[:cut_len]
    att_k_i = att_k_i[:cut_len]
    att_w_i = att_w_i[:cut_len]

    strokes = res1
    strokes[:, 1:] = np.cumsum(strokes[:, 1:], axis=0)
    minx, maxx = np.min(strokes[:, 1]), np.max(strokes[:, 1])
    miny, maxy = np.min(strokes[:, 2]), np.max(strokes[:, 2])
    split = split_strokes(strokes)
    for sp in split:
        axarr[0].plot(sp[:, 0], sp[:, 1])
    axarr[1].imshow(att_w_i.T)
    #axarr[1].plot(att_k_i.mean(axis=-1))
    plt.savefig("plot_results1.png")
    plt.clf()
    plt.close()

    res2 = numpy_sample_bernoulli_and_bivariate_gmm(mus_i, sigmas_i, corrs_i, coeffs_i, bernoullis_i,
            random_state=random_state)
    res2 = res2[:cut_len]
    att_k_i = att_k_i[:cut_len]
    att_w_i = att_w_i[:cut_len]

    f, axarr = plt.subplots(2, 1)
    strokes = res2
    strokes[:, 1:] = np.cumsum(strokes[:, 1:], axis=0)
    minx, maxx = np.min(strokes[:, 1]), np.max(strokes[:, 1])
    miny, maxy = np.min(strokes[:, 2]), np.max(strokes[:, 2])
    split = split_strokes(strokes)
    for sp in split:
        axarr[0].plot(sp[:, 0], sp[:, 1])
    axarr[1].imshow(att_w_i.T)
    #axarr[1].plot(att_k_i.mean(axis=-1))
    plt.savefig("plot_results2.png")
    from IPython import embed; embed(); raise ValueError()
"""

