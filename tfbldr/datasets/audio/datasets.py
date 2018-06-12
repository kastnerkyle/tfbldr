from .audio_tools import stft
from .audio_tools import linear_to_mel_weight_matrix
from .audio_tools import stft
from .audio_tools import iterate_invert_spectrogram
from .audio_tools import soundsc
from ..text.cleaning import text_to_sequence
from ..text.cleaning import sequence_to_text

from scipy.io import wavfile
import numpy as np
import os

class wavfile_caching_mel_tbptt_iterator(object):
    def __init__(self, wavfile_list, txtfile_list=None,
                 window_size=512,
                 window_step=128,
                 n_mel_filters=80,
                 lower_edge_hertz=125.0,
                 upper_edge_hertz=7800.0,
                 clean_names=["english_cleaners",],
                 cache_dir_base="/Tmp/kastner/tfbldr_cache",
                 shuffle=False, random_state=None):
         self.wavfile_list = wavfile_list
         self.txtfile_list = txtfile_list
         self.random_state = random_state
         self.shuffle = shuffle
         self.cache_dir_base = cache_dir_base
         self.lower_edge_hertz = lower_edge_hertz
         self.upper_edge_hertz = upper_edge_hertz
         self.clean_names = clean_names
         self.window_size = window_size
         self.window_step = window_step
         self.n_mel_filters = n_mel_filters

         if shuffle and random_state == None:
             raise ValueError("Must pass random_state in")
         if txtfile_list is not None:
             # try to match every txt file and every wav file by name
             txtbase = str(os.sep).join(self.txtfile_list[0].split(os.sep)[:-1])
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
             print("DONE")
             self.wavfile_list = wv_match
             self.txtfile_list = tx_match
         self.cache = self.cache_dir_base + os.sep + "-".join(self.wavfile_list[0].split(os.sep)[1:-1])
         if not os.path.exists(self.cache):
             os.makedirs(self.cache)
         wf = self.cache_read_wav_features(self.wavfile_list[0])
         txf = self.cache_read_txt_features(self.txtfile_list[0])
         from IPython import embed; embed(); raise ValueError()

    def calculate_log_mel_features(self, sample_rate, waveform, window_size, window_step, lower_edge_hertz, upper_edge_hertz, n_mel_filters):
        res = np.abs(stft(waveform, windowsize=window_size, step=window_step, real=False, compute_onesided=True))
        mels = linear_to_mel_weight_matrix(
            res.shape[1],
            sample_rate,
            lower_edge_hertz=lower_edge_hertz,
            upper_edge_hertz=min(float(sample_rate) // 2, upper_edge_hertz),
            n_filts=n_mel_filters, dtype=np.float64)
        mel_res = np.dot(res, mels)
        log_mel_res = np.log1p(mel_res)
        return log_mel_res

    def cache_read_wav_features(self, wavpath, return_npz=False):
        fname = ".".join(wavpath.split(os.sep)[-1].split(".")[:-1])
        if self.txtfile_list is not None:
            txtpart = "txt-clean{}".format(str("".join(self.clean_names)))
            melpart = "logmel-wsz{}-wst{}-leh{}-ueh{}-nmel{}.npz".format(self.window_size, self.window_step, int(self.lower_edge_hertz), int(self.upper_edge_hertz), self.n_mel_filters)
            npzpath = self.cache + os.sep + fname + txtpart + "-" + melpart
        else:
            npzpath = self.cache + os.sep + fname + "logmel-wsz{}-wst{}-leh{}-ueh{}-nmel{}.npz".format(self.window_size, self.window_step, int(self.lower_edge_hertz), int(self.upper_edge_hertz), self.n_mel_filters)
        if not os.path.exists(npzpath):
            sr, d = wavfile.read(wavpath)
            d = d.astype("float64")
            log_mels = self.calculate_log_mel_features(sr, d, self.window_size, self.window_step,
                                                       self.lower_edge_hertz, self.upper_edge_hertz, self.n_mel_filters)
            np.savez(npzpath, wavpath=wavpath, sample_rate=sr, log_mels=log_mels)
        npzfile = np.load(npzpath)
        log_mels = npzfile["log_mels"]
        if return_npz:
            return log_mels, npzfile, npzpath
        else:
            return log_mels

    def cache_read_txt_features(self, txtpath, npzfile=None, npzpath=None):
        if npzfile is None or "txt" not in npzfile:
            with open(txtpath, "rb") as f:
                lines = f.readlines()
            raw_txt = lines[0]
            clean_txt_ints = text_to_sequence(raw_txt, self.clean_names)
        from IPython import embed; embed(); raise ValueError()
        print("jdwajkdlwa")

    def cache_read_wav_and_txt_features(self, wavpath, txtpath):
        from IPython import embed; embed(); raise ValueError()

