from __future__ import print_function
# Author: Kyle Kastner
# License: BSD 3-clause
# Thanks to Jose (@sotelo) for tons of guidance and debug help
# Credit also to Junyoung (@jych) and Shawn (@shawntan) for help/utility funcs
import os
import re
import tarfile
from collections import Counter, OrderedDict
from bs4 import BeautifulSoup as Soup
import sys
import pickle
import numpy as np
import fnmatch
from scipy import linalg
from functools import wraps
import exceptions
import subprocess
import shutil
import xml
import xml.etree.cElementTree as ElementTree
import HTMLParser


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


class BatchGenerator(object):
    def __init__(self, batch_size, seq_len, random_seed):
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.random_state = np.random.RandomState(random_seed)
        dataset, labels, self.translation = self.load_dataset()
        ndataset, nlabels = [], []
        for i in range(len(dataset)):
            if len(dataset[i]) >= seq_len + 1:
                ndataset += [dataset[i]]
                nlabels += [labels[i]]
        del dataset, labels
        self.dataset, labels = ndataset, nlabels

        self.num_letters = len(self.translation)
        # pad all labels to be the same length
        max_len = max(map(lambda x: len(x), labels))
        self.labels = np.array([np.concatenate([np.eye(self.num_letters, dtype=np.float32)[l],
                                                np.zeros((max_len - len(l) + 1, self.num_letters),
                                                         dtype=np.float32)],
                                               axis=0)
                                for l in labels])
        self.max_len = self.labels.shape[1]
        self.indices = self.random_state.choice(len(self.dataset), size=(batch_size,), replace=False)
        self.batches = np.zeros((batch_size,), dtype=np.int32)

    def next_batch(self):
        coords = np.zeros((self.batch_size, self.seq_len + 1, 3), dtype=np.float32)
        sequence = np.zeros((self.batch_size, self.max_len, self.num_letters), dtype=np.float32)
        reset_states = np.ones((self.batch_size, 1), dtype=np.float32)
        needed = False
        for i in range(self.batch_size):
            if self.batches[i] + self.seq_len + 1 > self.dataset[self.indices[i]].shape[0]:
                ni = self.random_state.randint(0, len(self.dataset) - 1)
                self.indices[i] = ni
                self.batches[i] = 0
                reset_states[i] = 0.
                needed = True
            coords[i, :, :] = self.dataset[self.indices[i]][self.batches[i]: self.batches[i] + self.seq_len + 1]
            sequence[i] = self.labels[self.indices[i]]
            self.batches[i] += self.seq_len
        return coords, sequence, reset_states, needed

    def next_batch2(self):
        r = self.next_batch()
        coords = r[0].transpose(1, 0, 2)
        coords_mask = make_mask(coords)
        seq = r[1].transpose(1, 0, 2)
        seq_mask = make_mask(seq)
        reset = r[2]
        needed = r[3]
        return coords, coords_mask, seq, seq_mask, reset

    @staticmethod
    def load_dataset():
        d = fetch_iamondb()
        return d["data"], d["target"], d["vocabulary"]


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    labels_shape = labels_dense.shape
    labels_dense = labels_dense.reshape([-1])
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    labels_one_hot = labels_one_hot.reshape(labels_shape+(num_classes,))
    return labels_one_hot


def tokenize_ind(phrase, vocabulary):
    phrase = phrase + " "
    vocabulary_size = len(vocabulary.keys())
    phrase = [vocabulary[char_] for char_ in phrase]
    phrase = np.array(phrase, dtype='int32').ravel()
    phrase = dense_to_one_hot(phrase, vocabulary_size)
    return phrase


# https://mrcoles.com/blog/3-decorator-examples-and-awesome-python/
def rsync_fetch(fetch_func, machine_to_fetch_from, *args, **kwargs):
    """
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

