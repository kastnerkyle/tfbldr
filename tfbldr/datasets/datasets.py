from __future__ import print_function
# Author: Kyle Kastner
# License: BSD 3-clause
# Thanks to Jose (@sotelo) for tons of guidance and debug help
# Credit also to Junyoung (@jych) and Shawn (@shawntan) for help/utility funcs
import os
import re
import tarfile
from collections import Counter, OrderedDict
import sys
import pickle
import numpy as np
import fnmatch
from scipy import linalg
from functools import wraps
import exceptions
import subprocess
import copy
import shutil
import xml
import xml.etree.cElementTree as ElementTree
import HTMLParser
import functools
import operator
import gzip
import struct
import array

from ..core import download
from ..core import get_logger

logger = get_logger()


def get_tfbldr_dataset_dir(dirname=None):
    lookup_dir = os.getenv("TFBLDR_DATASETS", os.path.join(
        os.path.expanduser("~"), "tfbldr_datasets"))
    if not os.path.exists(lookup_dir):
        logger.info("TFBLDR_DATASETS directory {} not found, creating".format(lookup_dir))
        os.mkdir(lookup_dir)
    if dirname is None:
        return lookup_dir

    subdir = os.path.join(lookup_dir, dirname)
    if not os.path.exists(subdir):
        logger.info("TFBLDR_DATASETS subdirectory {} not found, creating".format(subdir))
        os.mkdir(subdir)
    return subdir


def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
        shutil.copystat(src, dst)
    lst = os.listdir(src)
    if ignore:
        excl = ignore(src, lst)
        lst = [x for x in lst if x not in excl]
    for item in lst:
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if symlinks and os.path.islink(s):
            if os.path.lexists(d):
                os.remove(d)
            os.symlink(os.readlink(s), d)
            try:
                st = os.lstat(s)
                mode = stat.S_IMODE(st.st_mode)
                os.lchmod(d, mode)
            except:
                pass  # lchmod not available
        elif os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def pwrap(args, shell=False):
    p = subprocess.Popen(args, shell=shell, stdout=subprocess.PIPE,
                         stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    return p

# Print output
# http://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
def execute(cmd, shell=False):
    popen = pwrap(cmd, shell=shell)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line

    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def pe(cmd, shell=False, verbose=True):
    """
    Print and execute command on system
    """
    all_lines = []
    for line in execute(cmd, shell=shell):
        if verbose:
            print(line, end="")
        all_lines.append(line.strip())
    return all_lines


def make_mask(arr):
    mask = np.ones_like(arr[:, :, 0])
    last_step = arr.shape[0] * arr[0, :, 0]
    for mbi in range(arr.shape[1]):
        for step in range(arr.shape[0]):
            if arr[step:, mbi].min() == 0. and arr[step:, mbi].max() == 0.:
                last_step[mbi] = step
                mask[step:, mbi] = 0.
                break
    return mask


# https://mrcoles.com/blog/3-decorator-examples-and-awesome-python/
def rsync_fetch(fetch_func, machine_to_fetch_from, *args, **kwargs):
    """
    assumes the filename in IOError is a subdir, will rsync one level above that

    be sure not to call it as
    rsync_fetch(fetch_func, machine_name)
    not
    rsync_fetch(fetch_func(), machine_name)
    """
    try:
        r = fetch_func(*args, **kwargs)
    except Exception as e:
        if isinstance(e, IOError):
            full_path = e.filename
            filedir = str(os.sep).join(full_path.split(os.sep)[:-1])
            if not os.path.exists(filedir):
                if filedir[-1] != "/":
                    fd = filedir + "/"
                else:
                    fd = filedir
                os.makedirs(fd)

            if filedir[-1] != "/":
                fd = filedir + "/"
            else:
                fd = filedir

            if not os.path.exists(full_path):
                sdir = str(machine_to_fetch_from) + ":" + fd
                cmd = "rsync -avhp --progress %s %s" % (sdir, fd)
                pe(cmd, shell=True)
        else:
            print("unknown error {}".format(e))
        r = fetch_func(*args, **kwargs)
    return r


def plot_lines_iamondb_example(X, title="", save_name=None):
    import matplotlib.pyplot as plt
    f, ax = plt.subplots()
    x = np.cumsum(X[:, 1])
    y = np.cumsum(X[:, 2])

    size_x = x.max() - x.min()
    size_y = y.max() - y.min()

    f.set_size_inches(5 * size_x / size_y, 5)
    cuts = np.where(X[:, 0] == 1)[0]
    start = 0

    for cut_value in cuts:
        ax.plot(x[start:cut_value], y[start:cut_value],
                'k-', linewidth=1.5)
        start = cut_value + 1
    ax.axis('equal')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_title(title)

    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
    plt.close()


def implot(arr, title="", cmap="gray", save_name=None):
    import matplotlib.pyplot as plt
    f, ax = plt.subplots()
    ax.matshow(arr, cmap=cmap)
    plt.axis("off")

    def autoaspect(x_range, y_range):
        """
        The aspect to make a plot square with ax.set_aspect in Matplotlib
        """
        mx = max(x_range, y_range)
        mn = min(x_range, y_range)
        if x_range <= y_range:
            return mx / float(mn)
        else:
            return mn / float(mx)

    x1 = arr.shape[0]
    y1 = arr.shape[1]
    asp = autoaspect(x1, y1)
    ax.set_aspect(asp)
    plt.title(title)
    if save_name is None:
        plt.show()
    else:
        plt.savefig(save_name)


"""
- all points:
>> [[x1, y1, e1], ..., [xn, yn, en]]
- indexed values
>> [h1, ... hn]
"""


def distance(p1, p2, axis=None):
    return np.sqrt(np.sum(np.square(p1 - p2), axis=axis))


def clear_middle(pts):
    to_remove = set()
    for i in range(1, len(pts) - 1):
        p1, p2, p3 = pts[i - 1: i + 2, :2]
        dist = distance(p1, p2) + distance(p2, p3)
        if dist > 1500:
            to_remove.add(i)
    npts = []
    for i in range(len(pts)):
        if i not in to_remove:
            npts += [pts[i]]
    return np.array(npts)


def separate(pts):
    seps = []
    for i in range(0, len(pts) - 1):
        if distance(pts[i], pts[i+1]) > 600:
            seps += [i + 1]
    return [pts[b:e] for b, e in zip([0] + seps, seps + [len(pts)])]


def iamondb_extract(partial_path):
    """
    Lightly modified from https://github.com/Grzego/handwriting-generation/blob/master/preprocess.py
    """
    data = []
    charset = set()

    file_no = 0
    pth = os.path.join(partial_path, "original")
    for root, dirs, files in os.walk(pth):
        # sort the dirs to iterate the same way every time
        # https://stackoverflow.com/questions/18282370/os-walk-iterates-in-what-order
        dirs.sort()
        for file in files:
            file_name, extension = os.path.splitext(file)
            if extension == '.xml':
                file_no += 1
                print('[{:5d}] File {} -- '.format(file_no, os.path.join(root, file)), end='')
                xml = ElementTree.parse(os.path.join(root, file)).getroot()
                transcription = xml.findall('Transcription')
                if not transcription:
                    print('skipped')
                    continue
                #texts = [html.unescape(s.get('text')) for s in transcription[0].findall('TextLine')]
                texts = [HTMLParser.HTMLParser().unescape(s.get('text')) for s in transcription[0].findall('TextLine')]
                points = [s.findall('Point') for s in xml.findall('StrokeSet')[0].findall('Stroke')]
                strokes = []
                mid_points = []
                for ps in points:
                    pts = np.array([[int(p.get('x')), int(p.get('y')), 0] for p in ps])
                    pts[-1, 2] = 1

                    pts = clear_middle(pts)
                    if len(pts) == 0:
                        continue

                    seps = separate(pts)
                    for pss in seps:
                        if len(seps) > 1 and len(pss) == 1:
                            continue
                        pss[-1, 2] = 1

                        xmax, ymax = max(pss, key=lambda x: x[0])[0], max(pss, key=lambda x: x[1])[1]
                        xmin, ymin = min(pss, key=lambda x: x[0])[0], min(pss, key=lambda x: x[1])[1]

                        strokes += [pss]
                        mid_points += [[(xmax + xmin) / 2., (ymax + ymin) / 2.]]
                distances = [-(abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]))
                             for p1, p2 in zip(mid_points, mid_points[1:])]
                splits = sorted(np.argsort(distances)[:len(texts) - 1] + 1)
                lines = []
                for b, e in zip([0] + splits, splits + [len(strokes)]):
                    lines += [[p for pts in strokes[b:e] for p in pts]]
                print('lines = {:4d}; texts = {:4d}'.format(len(lines), len(texts)))
                charset |= set(''.join(texts))
                data += [(texts, lines)]
    print('data = {}; charset = ({}) {}'.format(len(data), len(charset), ''.join(sorted(charset))))

    translation = {'<NULL>': 0}
    for c in ''.join(sorted(charset)):
        translation[c] = len(translation)

    def translate(txt):
        return list(map(lambda x: translation[x], txt))

    dataset = []
    labels = []
    for texts, lines in data:
        for text, line in zip(texts, lines):
            line = np.array(line, dtype=np.float32)
            line[:, 0] = line[:, 0] - np.min(line[:, 0])
            line[:, 1] = line[:, 1] - np.mean(line[:, 1])

            dataset += [line]
            labels += [translate(text)]

    whole_data = np.concatenate(dataset, axis=0)

    std_y = np.std(whole_data[:, 1])
    norm_data = []
    for line in dataset:
        line[:, :2] /= std_y
        norm_data += [line]
    dataset = norm_data

    print('datset = {}; labels = {}'.format(len(dataset), len(labels)))

    save_path = os.path.join(partial_path, 'preprocessed_data')
    try:
        os.makedirs(save_path)
    except FileExistsError:
        pass
    np.save(os.path.join(save_path, 'dataset'), np.array(dataset))
    np.save(os.path.join(save_path, 'labels'), np.array(labels))
    with open(os.path.join(save_path, 'translation.pkl'), 'wb') as file:
        pickle.dump(translation, file)
    print("Preprocessing finished and cached at {}".format(save_path))


def check_fetch_iamondb():
    """ Check for IAMONDB data

        This dataset cannot be downloaded automatically!
    """
    #partial_path = get_dataset_dir("iamondb")
    partial_path = os.sep + "Tmp" + os.sep + "kastner" + os.sep + "iamondb"
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    combined_data_path = os.path.join(partial_path, "original-xml-part.tar.gz")
    untarred_data_path = os.path.join(partial_path, "original")
    if not os.path.exists(combined_data_path):
        files = "original-xml-part.tar.gz"
        url = "http://www.iam.unibe.ch/fki/databases/"
        url += "iam-on-line-handwriting-database/"
        url += "download-the-iam-on-line-handwriting-database"
        err = "Path %s does not exist!" % combined_data_path
        err += " Download the %s files from %s" % (files, url)
        err += " and place them in the directory %s" % partial_path
        print("WARNING: {}".format(err))
    return partial_path


def fetch_iamondb():
    partial_path = check_fetch_iamondb()
    combined_data_path = os.path.join(partial_path, "original-xml-part.tar.gz")
    untarred_data_path = os.path.join(partial_path, "original")
    if not os.path.exists(untarred_data_path):
        print("Now untarring {}".format(combined_data_path))
        tar = tarfile.open(combined_data_path, "r:gz")
        tar.extractall(partial_path)
        tar.close()

    saved_dataset_path = os.path.join(partial_path, 'preprocessed_data')

    if not os.path.exists(saved_dataset_path):
        iamondb_extract(partial_path)

    dataset_path = os.path.join(saved_dataset_path, "dataset.npy")
    labels_path = os.path.join(saved_dataset_path, "labels.npy")
    translation_path = os.path.join(saved_dataset_path, "translation.pkl")

    dataset = np.load(dataset_path)
    dataset = [np.array(d) for d in dataset]

    temp = []
    for d in dataset:
        # dataset stores actual pen points, but we will train on differences between consecutive points
        offs = d[1:, :2] - d[:-1, :2]
        ends = d[1:, 2]
        temp += [np.concatenate([[[0., 0., 1.]], np.concatenate([offs, ends[:, None]], axis=1)], axis=0)]
    # because lines are of different length, we store them in python array (not numpy)
    dataset = temp
    labels = np.load(labels_path)
    labels = [np.array(l) for l in labels]
    with open(translation_path, 'rb') as f:
        translation = pickle.load(f)
    # be sure of consisten ordering
    new_translation = OrderedDict()
    for k in sorted(translation.keys()):
        new_translation[k] = translation[k]
    translation = new_translation
    dataset_storage = {}
    dataset_storage["data"] = dataset
    dataset_storage["target"] = labels
    inverse_translation = {v: k for k, v in translation.items()}
    dataset_storage["target_phrases"] = ["".join([inverse_translation[ci] for ci in labels[i]]) for i in range(len(labels))]
    dataset_storage["vocabulary_size"] = len(translation)
    dataset_storage["vocabulary"] = translation
    return dataset_storage


def check_fetch_ljspeech(conditioning_type):
    """ Check for ljspeech

        This dataset cannot be downloaded or preprocessed automatically!
    """
    if conditioning_type == "hybrid":
        partial_path = os.sep + "Tmp" + os.sep + "kastner" + os.sep + "lj_speech_hybrid_speakers"
    else:
        raise ValueError("Unknown conditioning_type={} specified".format(conditioning_type))
    if not os.path.exists(partial_path):
        os.makedirs(partial_path)
    if not os.path.exists(partial_path + os.sep + "norm_info") or not os.path.exists(partial_path + os.sep + "numpy_features"):
        err = "lj_speech_hybrid_speakers files not found. These files need special preprocessing! Do that, and place norm_info and numpy_features in {}"
        print("WARNING: {}".format(err.format(partial_path)))
    return partial_path


def fetch_ljspeech(conditioning_type="hybrid"):
    """
    only returns file paths, and metadata/conversion routines
    """
    partial_path = check_fetch_ljspeech(conditioning_type)
    features_path = os.path.join(partial_path, "numpy_features")
    norm_path = os.path.join(partial_path, "norm_info")
    if not os.path.exists(features_path) or not os.path.exists(norm_path):
        e = IOError("No feature files found in {}, under {}".format(partial_path, features_path), None, features_path)
        raise e

    feature_files = [features_path + os.sep + f for f in os.listdir(features_path)]
    if len(feature_files) == 0:
        e = IOError("No feature files found in {}, under {}".format(partial_path, features_path), None, features_path)
        raise e

    ljspeech_hybridset = [' ', '!', ',', '-', '.', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    translation = OrderedDict()
    for n, k in enumerate(ljspeech_hybridset):
        translation[k] = n

    dataset_storage = {}
    dataset_storage["file_paths"] = feature_files
    dataset_storage["vocabulary_size"] = len(ljspeech_hybridset)
    dataset_storage["vocabulary"] = translation
    return dataset_storage


class IdxDecodeError(ValueError):
    """Raised when an invalid idx file is parsed."""
    pass


def parse_idx(fd):
    """
    Parse an IDX file, and return it as a numpy array.
    From https://github.com/datapythonista/mnist

    Parameters
    ----------
    fd : file
        File descriptor of the IDX file to parse
    endian : str
        Byte order of the IDX file. See [1] for available options

    Returns
    -------
    data : numpy.ndarray
        Numpy array with the dimensions and the data in the IDX file
    1. https://docs.python.org/3/library/struct.html#byte-order-size-and-alignment
    """
    DATA_TYPES = {0x08: 'B',  # unsigned byte
                  0x09: 'b',  # signed byte
                  0x0b: 'h',  # short (2 bytes)
                  0x0c: 'i',  # int (4 bytes)
                  0x0d: 'f',  # float (4 bytes)
                  0x0e: 'd'}  # double (8 bytes)

    header = fd.read(4)
    if len(header) != 4:
        raise IdxDecodeError('Invalid IDX file, file empty or does not contain a full header.')

    zeros, data_type, num_dimensions = struct.unpack('>HBB', header)

    if zeros != 0:
        raise IdxDecodeError('Invalid IDX file, file must start with two zero bytes. '
                             'Found 0x%02x' % zeros)

    try:
        data_type = DATA_TYPES[data_type]
    except KeyError:
        raise IdxDecodeError('Unknown data type 0x%02x in IDX file' % data_type)

    dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                    fd.read(4 * num_dimensions))

    data = array.array(data_type, fd.read())
    data.byteswap()  # looks like array.array reads data as little endian

    expected_items = functools.reduce(operator.mul, dimension_sizes)
    if len(data) != expected_items:
        raise IdxDecodeError('IDX file has wrong number of items. '
                             'Expected: %d. Found: %d' % (expected_items, len(data)))

    return np.array(data).reshape(dimension_sizes)


def check_fetch_mnist():
    mnist_dir = get_tfbldr_dataset_dir("mnist")
    base = "http://yann.lecun.com/exdb/mnist/"
    zips = [base + "train-images-idx3-ubyte.gz",
            base + "train-labels-idx1-ubyte.gz",
            base + "t10k-images-idx3-ubyte.gz",
            base + "t10k-labels-idx1-ubyte.gz"]

    for z in zips:
        fname = z.split("/")[-1]
        full_path = os.path.join(mnist_dir, fname)
        if not os.path.exists(full_path):
            logger.info("{} not found, downloading...".format(full_path))
            download(z, full_path)
    return mnist_dir


def fetch_mnist():
    """
    Flattened or image-shaped 28x28 mnist digits with pixel values in [0 - 1]

    n_samples : 70000
    n_feature : 784

    Returns
    -------
    summary : dict
        A dictionary cantaining data and image statistics.

        summary["data"] : float32 array, shape (70000, 784)
        summary["target"] : int32 array, shape (70000,)
        summary["images"] : float32 array, shape (70000, 28, 28, 1)
        summary["train_indices"] : int32 array, shape (50000,)
        summary["valid_indices"] : int32 array, shape (10000,)
        summary["test_indices"] : int32 array, shape (10000,)

    """
    data_path = check_fetch_mnist()
    train_image_gz = "train-images-idx3-ubyte.gz"
    train_label_gz = "train-labels-idx1-ubyte.gz"
    test_image_gz = "t10k-images-idx3-ubyte.gz"
    test_label_gz = "t10k-labels-idx1-ubyte.gz"

    out = []
    for path in [train_image_gz, train_label_gz, test_image_gz, test_label_gz]:
        f = gzip.open(os.path.join(data_path, path), 'rb')
        out.append(parse_idx(f))
        f.close()
    train_indices = np.arange(0, 50000)
    valid_indices = np.arange(50000, 60000)
    test_indices = np.arange(60000, 70000)
    data = np.concatenate((out[0], out[2]),
                          axis=0).astype(np.float32)
    target = np.concatenate((out[1], out[3]),
                            axis=0).astype(np.int32)
    return {"data": copy.deepcopy(data.reshape((data.shape[0], -1))),
            "target": target,
            "images": data[..., None],
            "train_indices": train_indices.astype(np.int32),
            "valid_indices": valid_indices.astype(np.int32),
            "test_indices": test_indices.astype(np.int32)}


class list_iterator(object):
    def __init__(self, list_of_iteration_args, batch_size,
                 one_hot_size=None, random_state=None):
        """
        one_hot_size
        should be either None, or a list of one hot size desired
        same length as list_of_iteration_args

        list_of_iteration_args = [my_image_data, my_label_data]
        one_hot_size = [None, 10]
        """
        self.list_of_iteration_args = list_of_iteration_args
        self.batch_size = batch_size

        self.random_state = random_state
        if random_state is None:
            raise ValueError("Must pass random state for random selection")

        self.one_hot_size = one_hot_size
        if one_hot_size is not None:
            assert len(one_hot_size) == len(list_of_iteration_args)

        iteration_args_lengths = []
        iteration_args_dims = []
        for n, ts in enumerate(list_of_iteration_args):
            c = [(li, np.array(tis).shape) for li, tis in enumerate(ts)]
            if len(iteration_args_lengths) > 0:
                assert c[-1][0] == iteration_args_lengths[-1]
                assert c[-1][1] == iteration_args_dims[-1]
            iteration_args_lengths.append(c[-1][0] + 1)
            iteration_args_dims.append(c[-1][1])
        self.iteration_args_lengths_ = iteration_args_lengths
        self.iteration_args_dims_ = iteration_args_dims

        # set up the matrices to slice one_hot indexes out of
        # todo: setup slice functions? or just keep handling in next_batch
        if one_hot_size is None:
            self._oh_slicers = [None] * len(list_of_iteration_args)
        else:
            self._oh_slicers = []
            for ooh in one_hot_size:
                if ooh is None:
                    self._oh_slicers.append(None)
                else:
                    self._oh_slicers.append(np.eye(ooh, dtype=np.float32))

        # set up the indices selected for the first batch
        self.indices_ = self.random_state.choice(self.iteration_args_lengths_[0],
                                                 size=(batch_size,), replace=False)

    def next_batch(self):
        # whether the result is "fresh" or continuation
        next_batches = []
        for l in range(len(self.list_of_iteration_args)):
            if self._oh_slicers[l] is None:
                t = np.zeros([self.batch_size] + list(self.iteration_args_dims_[l]), dtype=np.float32)
            else:
                t = np.zeros([self.batch_size] + list(self.iteration_args_dims_[l])[:-1] + [self._oh_slicers[l].shape[-1]], dtype=np.float32)
            next_batches.append(t)
        self.indices_ = self.random_state.choice(self.iteration_args_lengths_[0],
                                                 size=(self.batch_size,), replace=False)
        return next_batches


class tbptt_list_iterator(object):
    def __init__(self, tbptt_seqs, list_of_other_seqs, batch_size,
                 truncation_length,
                 tbptt_one_hot_size=None, other_one_hot_size=None,
                 random_state=None):
        """
        skips sequences shorter than truncation_len
        also cuts the tail off

        tbptt_one_hot_size
        should be either None, or the one hot size desired

        other_one_hot_size
        should either be None (if not doing one-hot) or a list the same length
        as the respective argument with integer one hot size, or None
        for no one_hot transformation, example:

        list_of_other_seqs = [my_char_data, my_vector_data]
        other_one_hot_size = [127, None]
        """
        self.tbptt_seqs = tbptt_seqs
        self.list_of_other_seqs = list_of_other_seqs
        self.batch_size = batch_size
        self.truncation_length = truncation_length

        self.random_state = random_state
        if random_state is None:
            raise ValueError("Must pass random state for random selection")

        self.tbptt_one_hot_size = tbptt_one_hot_size

        self.other_one_hot_size = other_one_hot_size
        if other_one_hot_size is not None:
            assert len(other_one_hot_size) == len(list_of_other_seqs)

        tbptt_seqs_length = [n for n, i in enumerate(tbptt_seqs)][-1] + 1
        self.indices_lookup_ = {}
        s = 0
        for n, ts in enumerate(tbptt_seqs):
            if len(ts) >= truncation_length + 1:
                self.indices_lookup_[s] = n
                s += 1

        # this one has things removed
        self.tbptt_seqs_length_ = len(self.indices_lookup_)

        other_seqs_lengths = []
        for other_seqs in list_of_other_seqs:
            r = [n for n, i in enumerate(other_seqs)]
            l = r[-1] + 1
            other_seqs_lengths.append(l)
        self.other_seqs_lengths_ = other_seqs_lengths

        other_seqs_max_lengths = []
        for other_seqs in list_of_other_seqs:
            max_l = -1
            for os in other_seqs:
                max_l = len(os) if len(os) > max_l else max_l
            other_seqs_max_lengths.append(max_l)
        self.other_seqs_max_lengths_ = other_seqs_max_lengths

        # make sure all sequences have the minimum number of elements
        base = self.tbptt_seqs_length_
        for sl in self.other_seqs_lengths_:
            assert sl >= base

        # set up the matrices to slice one_hot indexes out of
        # todo: setup slice functions? or just keep handling in next_batch
        if tbptt_one_hot_size is None:
            self._tbptt_oh_slicer = None
        else:
            self._tbptt_oh_slicer = np.eye(tbptt_one_hot_size)

        if other_one_hot_size is None:
            self._other_oh_slicers = [None] * len(other_seq_lengths)
        else:
            self._other_oh_slicers = []
            for ooh in other_one_hot_size:
                if ooh is None:
                    self._other_oh_slicers.append(None)
                else:
                    self._other_oh_slicers.append(np.eye(ooh, dtype=np.float32))
        # set up the indices selected for the first batch
        self.indices_ = np.array([self.indices_lookup_[si]
                                  for si in self.random_state.choice(self.tbptt_seqs_length_,
                                      size=(batch_size,), replace=False)])
        # set up the batch offset indicators for tracking where we are
        self.batches_ = np.zeros((batch_size,), dtype=np.int32)

    def next_batch(self):
        # whether the result is "fresh" or continuation
        reset_states = np.ones((self.batch_size, 1), dtype=np.float32)
        for i in range(self.batch_size):
            # cuts off the end of every long sequence! tricky logic
            if self.batches_[i] + self.truncation_length + 1 > self.tbptt_seqs[self.indices_[i]].shape[0]:
                ni = self.indices_lookup_[self.random_state.randint(0, self.tbptt_seqs_length_ - 1)]
                self.indices_[i] = ni
                self.batches_[i] = 0
                reset_states[i] = 0.

        # could slice before one hot to be slightly more efficient but eh
        items = [self.tbptt_seqs[ii] for ii in self.indices_]
        if self._tbptt_oh_slicer is None:
            truncation_items = items
        else:
            truncation_items = [self._tbptt_oh_slicer[ai] for ai in items]

        other_items = []
        for oi in range(len(self.list_of_other_seqs)):
            items = [self.list_of_other_seqs[oi][ii] for ii in self.indices_]
            if self._other_oh_slicers[oi] is None:
                other_items.append(items)
            else:
                other_items.append([self._other_oh_slicers[oi][ai] for ai in items])

        # make storage
        tbptt_arr = np.zeros((self.truncation_length + 1, self.batch_size, truncation_items[0].shape[-1]), dtype=np.float32)
        other_arrs = [np.zeros((self.other_seqs_max_lengths_[ni], self.batch_size, other_arr[0].shape[-1]), dtype=np.float32)
                      for ni, other_arr in enumerate(other_items)]
        for i in range(self.batch_size):
            ns = truncation_items[i][self.batches_[i]:self.batches_[i] + self.truncation_length + 1]
            # dropped sequences shorter than truncation_len already
            tbptt_arr[:, i, :] = ns
            for na, oa in enumerate(other_arrs):
                oa[:len(other_items[na][i]), i, :] = other_items[na][i]
            self.batches_[i] += self.truncation_length
        return [tbptt_arr,] + other_arrs + [reset_states,]

    def next_masked_batch(self):
        r = self.next_batch()
        # reset is the last element
        end_result = []
        for ri in r[:-1]:
            ri_mask = make_mask(ri)
            end_result.append(ri)
            end_result.append(ri_mask)
        end_result.append(r[-1])
        return end_result


class tbptt_file_list_iterator(object):
    def __init__(self, list_of_files,
                 file_seqs_access_fn,
                 batch_size,
                 truncation_length,
                 tbptt_one_hot_size=None,
                 other_one_hot_size=None,
                 random_state=None):
        """
        skips sequences shorter than truncation_len
        also cuts the tail off

        tbptt_one_hot_size
        should be either None, or the one hot size desired

        other_one_hot_size
        should either be None (if not doing one-hot) or a list the same length
        as the other_seqs returned from file_seqs_access_fn with integer one hot size, or None
        for no one_hot transformation, example:

        list_of_other_seqs = [my_char_data, my_vector_data]
        other_one_hot_size = [127, None]
        """
        self.list_of_files = list_of_files
        # gets a file path, returns (tbptt_seq, other_seqs)
        # if one_hot, the respective elements need to be *indices*
        self.file_seqs_access_fn = file_seqs_access_fn
        self.batch_size = batch_size
        self.truncation_length = truncation_length

        self.random_state = random_state
        if random_state is None:
            raise ValueError("Must pass random state for random selection")

        self.tbptt_one_hot_size = tbptt_one_hot_size
        if self.tbptt_one_hot_size is None:
            self._tbptt_oh_slicer = None
        else:
            self._tbptt_oh_slicer = np.eye(tbptt_one_hot_size)

        self.other_one_hot_size = other_one_hot_size
        if other_one_hot_size is None:
            self._other_oh_slicers = [None] * 20 # if there's more than 20 of these we have a problem
        else:
            self._other_oh_slicers = []
            for ooh in other_one_hot_size:
                if ooh is None:
                    self._other_oh_slicers.append(None)
                else:
                    self._other_oh_slicers.append(np.eye(ooh, dtype=np.float32))

        self.indices_ = self.random_state.choice(len(self.list_of_files), size=(batch_size,), replace=False)
        self.batches_ = np.zeros((batch_size,), dtype=np.float32)
        self.current_fnames_ = None

        fnames = [self.list_of_files[i] for i in self.indices_]
        self.current_fnames_ = fnames
        datas = [self.file_seqs_access_fn(f) for f in fnames]
        tbptt_seqs = [d[0] for d in datas]
        other_seqs = [d[1:] for d in datas]

        self.current_tbptt_seqs_ = []
        self.current_other_seqs_ = []

        for idx in range(len(tbptt_seqs)):
            if not (len(tbptt_seqs[idx]) >= self.truncation_length + 1):
                new_tbptt = tbptt_seqs[idx]
                new_others = other_seqs[idx]
                num_tries = 0
                while not (len(new_tbptt) >= self.truncation_length + 1):
                    #print("idx {}:file {} too short, resample".format(idx, self.indices_[idx]))
                    new_file_idx = self.random_state.randint(0, len(self.list_of_files) - 1)
                    fname = self.list_of_files[new_file_idx]
                    new_data = self.file_seqs_access_fn(fname)
                    new_tbptt = new_data[0]
                    new_others = new_data[1:]
                    num_tries += 1
                    if num_tries >= 20:
                        raise ValueError("Issue in file iterator next_batch, can't get a large enough file after 20 tries!")
                self.indices_[idx] = new_file_idx
                tbptt_seqs[idx] = new_tbptt
                other_seqs[idx] = new_others
            self.current_tbptt_seqs_.append(tbptt_seqs[idx])
            self.current_other_seqs_.append(other_seqs[idx])

    def next_batch(self):
        reset_states = np.ones((self.batch_size, 1), dtype=np.float32)
        # check lengths and if it's too short, resample...
        for i in range(self.batch_size):
            if self.batches_[i] + self.truncation_length + 1 > len(self.current_tbptt_seqs_[i]):
                ni = self.random_state.randint(0, len(self.list_of_files) - 1)
                fname = self.list_of_files[ni]
                new_data = self.file_seqs_access_fn(fname)
                new_tbptt = new_data[0]
                new_others = new_data[1:]
                num_tries = 0
                while not (len(new_tbptt) >= self.truncation_length + 1):
                    ni = self.random_state.randint(0, len(self.list_of_files) - 1)
                    fname = self.list_of_files[ni]
                    new_data = self.file_seqs_access_fn(fname)
                    new_tbptt = new_data[0]
                    new_others = new_data[1:]
                    num_tries += 1
                    if num_tries >= 20:
                        print("Issue in file iterator next_batch, can't get a large enough file after {} tries! Tried {}, name {}".format(num_tries), ni, self.list_of_files[ni])
                self.batches_[i] = 0.
                reset_states[i] = 0.
                self.current_tbptt_seqs_[i] = new_tbptt
                self.current_other_seqs_[i] = new_others

        items = [self.current_tbptt_seqs_[ii] for ii in range(len(self.current_tbptt_seqs_))]
        if self._tbptt_oh_slicer is None:
            truncation_items = items
        else:
            truncation_items = [self._tbptt_oh_slicer[ai] for ai in items]

        other_items = []
        # batch index
        for oi in range(len(self.current_other_seqs_)):
            items = self.current_other_seqs_[oi]
            subitems = []
            for j in range(len(items)):
                if self._other_oh_slicers[j] is None:
                    subitems.append(np.array(items))
                else:
                    subitems.append(np.array([self._other_oh_slicers[j][ai] for ai in items[j]]))
            other_items.append(subitems)

        tbptt_arr = np.zeros((self.truncation_length + 1, self.batch_size, truncation_items[0].shape[-1]))
        other_seqs_max_lengths = [max([len(other_items[i][j]) for i in range(len(other_items))])
                                       for j in range(len(other_items[i]))]
        other_arrs = [np.zeros((other_seqs_max_lengths[ni], self.batch_size, np.array(other_items[0][ni]).shape[-1]), dtype=np.float32)
                      for ni in range(len(other_items[0]))]

        for i in range(self.batch_size):
            ns = truncation_items[i][int(self.batches_[i]):int(self.batches_[i] + self.truncation_length + 1)]
            tbptt_arr[:, i, :] = ns
            for na in range(len(other_arrs)):
                other_arrs[na][:len(other_items[i][na]), i, :] = other_items[i][na]
            self.batches_[i] += self.truncation_length
        return [tbptt_arr,] + other_arrs + [reset_states,]


    def next_masked_batch(self):
        r = self.next_batch()
        # reset is the last element
        end_result = []
        for ri in r[:-1]:
            ri_mask = make_mask(ri)
            end_result.append(ri)
            end_result.append(ri_mask)
        end_result.append(r[-1])
        return end_result
