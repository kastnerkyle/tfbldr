from .audio_tools import stft
from .audio_tools import linear_to_mel_weight_matrix
from .audio_tools import stft
from .audio_tools import iterate_invert_spectrogram
from .audio_tools import soundsc
from ..text.cleaning import text_to_sequence
from ..text.cleaning import sequence_to_text
from ..text.cleaning import get_vocabulary_size

from scipy.io import wavfile
import numpy as np
import copy
import os


class wavfile_caching_mel_tbptt_iterator(object):
    def __init__(self, wavfile_list, txtfile_list,
                 batch_size,
                 truncation_length,
                 wav_scale = 2 ** 15,
                 window_size=512,
                 window_step=128,
                 n_mel_filters=80,
                 mel_min=0.01,
                 mel_max=10000.,
                 return_normalized=True,
                 lower_edge_hertz=125.0,
                 upper_edge_hertz=7800.0,
                 start_index=0,
                 stop_index=None,
                 clean_names=["english_cleaners",],
                 cache_dir_base="/Tmp/kastner/tfbldr_cache",
                 shuffle=False, random_state=None):
         self.wavfile_list = wavfile_list
         self.wav_scale = wav_scale
         self.txtfile_list = txtfile_list
         self.batch_size = batch_size
         self.truncation_length = truncation_length
         self.random_state = random_state
         self.shuffle = shuffle
         self.cache_dir_base = cache_dir_base
         self.mel_min = mel_min
         self.mel_max = mel_max
         self.log_mel_min_ = 20. * np.log10(mel_min)
         self.log_mel_max_ = 20. * np.log10(mel_max)
         self.return_normalized = return_normalized
         self.lower_edge_hertz = lower_edge_hertz
         self.upper_edge_hertz = upper_edge_hertz
         self.clean_names = clean_names
         self.vocabulary_size = get_vocabulary_size(clean_names)
         self.window_size = window_size
         self.window_step = window_step
         self.n_mel_filters = n_mel_filters
         self.start_index = start_index
         self.stop_index = stop_index

         if shuffle and random_state == None:
             raise ValueError("Must pass random_state in")
         if txtfile_list is not None:
             # try to match every txt file and every wav file by name
             wv_bases = sorted([str(os.sep).join(wv.split(os.sep)[:-1]) for wv in self.wavfile_list])
             tx_bases = sorted([str(os.sep).join(tx.split(os.sep)[:-1]) for tx in self.txtfile_list])
             wv_names = sorted([wv.split(os.sep)[-1] for wv in self.wavfile_list])
             tx_names = sorted([tx.split(os.sep)[-1] for tx in self.txtfile_list])
             wv_i = 0
             tx_i = 0
             wv_match = []
             tx_match = []
             while True:
                 if tx_i >= len(tx_names) or wv_i >= len(wv_names):
                     break
                 if "." in tx_names[tx_i]:
                     tx_part = ".".join(tx_names[tx_i].split(".")[:1])
                 else:
                     # support txt files with no ext
                     tx_part = tx_names[tx_i]
                 wv_part = ".".join(wv_names[wv_i].split(".")[:1])
                 if wv_part == tx_part:
                     wv_match.append(wv_bases[wv_i] + os.sep + wv_names[wv_i])
                     tx_match.append(tx_bases[tx_i] + os.sep + tx_names[tx_i])
                     wv_i += 1
                     tx_i += 1
                 else:
                     print("WAV AND TXT NAMES DIDN'T MATCH AT STEP, ADD LOGIC")
                     from IPython import embed; embed(); raise ValueError()
             self.wavfile_list = wv_match
             self.txtfile_list = tx_match
         self.cache = self.cache_dir_base + os.sep + "-".join(self.wavfile_list[0].split(os.sep)[1:-1])
         if not os.path.exists(self.cache):
             os.makedirs(self.cache)

         if 0 < self.start_index < 1:
             self.start_index = int(len(self.wavfile_list) * self.start_index)
         elif self.start_index >= 1:
             self.start_index = int(self.start_index)
             if self.start_index >= len(self.wavfile_list):
                 raise ValueError("start_index {} >= length of wavfile list {}".format(self.start_index, len(self.wavfile_list)))
         elif self.start_index == 0:
             self.start_index = int(self.start_index)
         else:
             raise ValueError("Invalid value for start_index : {}".format(self.start_index))

         if self.stop_index == None:
             self.stop_index = len(self.wavfile_list)
         elif 0 < self.stop_index < 1:
             self.stop_index = int(len(self.wavfile_list) * self.stop_index)
         elif self.stop_index >= 1:
             self.stop_index = int(self.stop_index)
             if self.stop_index >= len(self.wavfile_list):
                 raise ValueError("stop_index {} >= length of wavfile list {}".format(self.stop_index, len(self.wavfile_list)))
         else:
             raise ValueError("Invalid value for stop_index : {}".format(self.stop_index))

         # set up for train / test splits
         self.all_indices_ = np.arange(len(self.wavfile_list))
         self.random_state.shuffle(self.all_indices_)
         self.all_indices_ = sorted(self.all_indices_[self.start_index:self.stop_index])

         self.current_indices_ = [self.random_state.choice(self.all_indices_) for i in range(self.batch_size)]
         self.current_offset_ = [0] * self.batch_size
         self.current_read_ = [self.cache_read_wav_and_txt_features(self.wavfile_list[i], self.txtfile_list[i]) for i in self.current_indices_]


    def next_batch(self):
        mel_batch = np.zeros((self.truncation_length, self.batch_size, self.n_mel_filters))
        resets = np.zeros((self.batch_size, 1))
        texts = []
        for bi in range(self.batch_size):
            wf, txf = self.current_read_[bi]
            trunc = self.current_offset_[bi] + self.truncation_length
            if len(wf) < self.truncation_length or trunc >= len(wf):
                if trunc >= len(wf):
                    resets[bi] = 1
                # if it's too short, drop entirely
                while True:
                    self.current_indices_[bi] = self.random_state.choice(self.all_indices_)
                    self.current_offset_[bi] = 0
                    try:
                        self.current_read_[bi] = self.cache_read_wav_and_txt_features(self.wavfile_list[self.current_indices_[bi]], self.txtfile_list[self.current_indices_[bi]])
                    except:
                        print("FILE / TEXT READ ERROR")
                        from IPython import embed; embed(); raise ValueError()
                    wf, txf = self.current_read_[bi]
                    if len(wf) > self.truncation_length:
                        break
            trunc = self.current_offset_[bi] + self.truncation_length
            wf_sub = wf[self.current_offset_[bi]:trunc]
            self.current_offset_[bi] = trunc
            mel_batch[:, bi] = wf_sub
            texts.append(txf)
        mlen = max([len(t) for t in texts])
        text_batch = np.zeros((mlen, self.batch_size, 1))
        for bi, txt in enumerate(texts):
            text_batch[:len(txt), bi, 0] = txt
        return mel_batch, text_batch, resets

    def next_masked_batch(self):
        m, t, r = self.next_batch()
        m_mask = np.ones_like(m[..., 0])
        t_mask = np.zeros_like(t[..., 0])
        t_mask[t[..., 0] > 0] = 1.
        return m, m_mask, t, t_mask, r

    def calculate_log_mel_features(self, sample_rate, waveform, window_size, window_step, lower_edge_hertz, upper_edge_hertz, n_mel_filters):
        res = np.abs(stft(waveform, windowsize=window_size, step=window_step, real=False, compute_onesided=True))
        mels = linear_to_mel_weight_matrix(
            res.shape[1],
            sample_rate,
            lower_edge_hertz=lower_edge_hertz,
            upper_edge_hertz=min(float(sample_rate) // 2, upper_edge_hertz),
            n_filts=n_mel_filters, dtype=np.float64)
        mel_res = np.dot(res, mels)
        mel_res[mel_res < self.mel_min] = self.mel_min
        mel_res[mel_res > self.mel_max] = self.mel_max
        log_mel_res = 20. * np.log10(mel_res)
        return log_mel_res

    def cache_read_wav_features(self, wavpath, return_npz=False):
        fname = ".".join(wavpath.split(os.sep)[-1].split(".")[:-1])
        if self.txtfile_list is not None:
            txtpart = "txt-clean{}".format(str("".join(self.clean_names)))
            melpart = "logmel-wsz{}-wst{}-mn{}-mx{}-leh{}-ueh{}-nmel{}.npz".format(self.window_size, self.window_step, self.mel_min, self.mel_max, int(self.lower_edge_hertz), int(self.upper_edge_hertz), self.n_mel_filters)
            npzpath = self.cache + os.sep + fname + txtpart + "-" + melpart
        else:
            npzpath = self.cache + os.sep + fname + "logmel-wsz{}-wst{}-leh{}-ueh{}-nmel{}.npz".format(self.window_size, self.window_step, int(self.lower_edge_hertz), int(self.upper_edge_hertz), self.n_mel_filters)
        if not os.path.exists(npzpath):
            sr, d = wavfile.read(wavpath)
            d = d.astype("float64")
            d = d / float(self.wav_scale)
            log_mels = self.calculate_log_mel_features(sr, d, self.window_size, self.window_step,
                                                       self.lower_edge_hertz, self.upper_edge_hertz, self.n_mel_filters)
            np.savez(npzpath, wavpath=wavpath, sample_rate=sr, log_mels=log_mels)
        npzfile = np.load(npzpath)
        log_mels = npzfile["log_mels"]
        if self.return_normalized:
            log_mels = (log_mels - self.log_mel_min_) / float(self.log_mel_max_ - self.log_mel_min_)
            log_mels = 2. * log_mels - 1.
        if return_npz:
            return log_mels, npzfile, npzpath
        else:
            return log_mels

    def cache_read_txt_features(self, txtpath, npzfile=None, npzpath=None):
        if npzfile is None or "raw_txt" not in npzfile:
            with open(txtpath, "rb") as f:
                lines = f.readlines()
            raw_txt = lines[0]
            int_txt = text_to_sequence(raw_txt, self.clean_names)
            clean_txt = sequence_to_text(int_txt)
            if npzfile is not None and "raw_txt" not in npzfile:
                d = {k: v for k, v in npzfile.items()}
                npzfile.close()
                d["raw_txt"] = raw_txt
                d["clean_txt"] = clean_txt
                d["int_txt"] = int_txt
                d["cleaners"] = "+".join(self.clean_names)
                np.savez(npzpath, **d)
        npzfile = np.load(npzpath)
        int_txt = npzfile["int_txt"]
        return int_txt

    def cache_read_wav_and_txt_features(self, wavpath, txtpath):
        wavfeats, npzfile, npzpath = self.cache_read_wav_features(wavpath, return_npz=True)
        txtfeats = self.cache_read_txt_features(txtpath, npzfile=npzfile, npzpath=npzpath)
        npzfile.close()
        return wavfeats, txtfeats
