import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy

import os
import numpy as np
from collections import Counter
from tfbldr.datasets import notes_to_midi
from tfbldr.datasets import midi_to_notes

basedir = "/u/kastner/music_npz_jos"

"""
cnt = Counter()
for fnpz in sorted(os.listdir(basedir)):
    print(fnpz)
    d = np.load(basedir + os.sep + fnpz)
    if len(d['centered']) < 1:
        print(fnpz + " had zero length")
        continue
    for mi in range(len(d['centered'])):
        measure = d['centered'][mi]
        cnt.update(measure.ravel())
"""

all_filenames = []
all_measurenums = []
all_piano_rolls = []
all_pitch_duration = []
all_keyframes = []
all_indexed = []
all_keys = []
all_modes = []
all_scalenotes = []

for fnpz in sorted(os.listdir(basedir)):
    print(fnpz)
    try:
        d = np.load(basedir + os.sep + fnpz)
    except:
        print("Unable to load {}, continuing".format(fnpz))

    if len(d["centered"]) < 1 or 'keyname' not in d:
        print(fnpz + " had zero length or no key")
        continue

    prs = copy.deepcopy(d["piano_rolls"])
    pds = copy.deepcopy(d["pitch_duration"])
    key = d["keyname"]
    mode = d["keymode"]
    notes = d["keynotes"]

    scale_lu = {}
    scale_lu["R"] = 0

    ordered_scale = ["R"]
    counter = 1
    for octave in ["1", "2", "3", "4", "5"]:
        for note in notes:
            ordered_scale.append(note + octave)
            scale_lu[note + octave] = counter
            counter += 1
    norm_lu = {v: k for k, v in scale_lu.items()}

    notes_lu = {os: notes_to_midi([[os]])[0][0] for os in ordered_scale}
    notes_lu["R"] = notes_to_midi([["R"]])[0][0]
    midi_lu = {v: k for k, v in notes_lu.items()}

    filename = fnpz
    keyframe_lu = {v: k for k, v in enumerate(np.arange(-13, 14 + 1))}
    diff_lu = {v: k for k, v in keyframe_lu.items()}

    measurenums = []
    keyframes = []
    indexed = []
    keys = []
    modes = []
    scalenotes = []

    last_non_rest = [0, 0, 0, 0]
    for n in range(len(prs)):
        # key and mode delta normalized repr
        # 0 rest, 1:28 is [-13, 14]
        pr_i = prs[n]
        pd_i = pds[n]
        if pr_i.shape[-1] != 4:
            #print("3 voices, skip for now")
            continue

        if pr_i.shape[0] != 48:
            new_pr_i = np.zeros((48, pr_i.shape[-1]))
            if pr_i.shape[0] == 32:
                # 32 into 48 is 4 into 6
                ii = 0
                oi = 0
                while True:
                    nt = pr_i[ii:ii + 4]
                    for v in range(pr_i.shape[-1]):
                        if len(np.unique(nt[:, v])) != 1:
                            if len(np.unique(nt[:, v])) == 2:
                                mn = np.min(nt[:, v])
                                mx = np.max(nt[:, v])
                                if np.sum(nt[:, v] == mn) == 2:
                                    nt[:, v] = mn
                                else:
                                    nt[:, v] = mx
                            else:
                                print("note changed :|")
                                from IPython import embed; embed(); raise ValueError()
                    new_pr_i[oi:oi + 6] = nt[0][None] # ii:ii + 3 all forced the same above
                    oi = oi + 6
                    ii = ii + 4
                    if ii >= 32:
                        break
                pr_i = new_pr_i
            else:
                #print("not length 48, needs normalization")
                continue

        loop_reset = False
        for unote in np.unique(pr_i):
            if unote not in midi_lu:
                #print("note not in key!")
                last_non_rest = [0, 0, 0, 0]
                loop_reset = True
                break
        if loop_reset:
            continue

        for v in range(pr_i.shape[-1]):
            non_rest = pr_i[pr_i[:, v] > 0, v]
            if last_non_rest[v] == 0:
                if len(non_rest) > 0:
                    last_non_rest[v] = scale_lu[midi_lu[non_rest[0]]]

        # put the midi notes in
        out = np.zeros_like(pr_i)
        for unote in np.unique(pr_i):
            out[pr_i == unote] = scale_lu[midi_lu[unote]]

        # calculate their offset relative to the keyframe
        # should become - whatever , 0 rest, 1 whatever where 1 is "same as keyframe"
        for v in range(pr_i.shape[-1]):
            subs = out[out[:, v] > 0, v]
            # shift the positive ones up, the others will be negative
            subs[subs >= last_non_rest[v]] += 1
            subs -= last_non_rest[v]
            out[out[:, v] > 0, v] = subs

        loop_reset = False
        for uni in np.unique(out):
            if uni not in keyframe_lu:
                #print("note not in key!")
                last_non_rest = [0, 0, 0, 0]
                loop_reset = True
                break
        if loop_reset:
            continue

        # finally, give each of these an index value so we can softmax it
        final = np.zeros_like(pr_i)
        for uni in np.unique(out):
            final[out == uni] = keyframe_lu[uni]

        indexed.append(final)
        keyframes.append(copy.deepcopy(last_non_rest))
        measurenums.append(n)

        for v in range(pr_i.shape[-1]):
            non_rest = pr_i[pr_i[:, v] > 0, v]
            if len(non_rest) > 0:
                last_non_rest[v] = scale_lu[midi_lu[non_rest[-1]]]

    filenames = [fnpz] * len(indexed)
    keys = [key] * len(indexed)
    modes = [mode] * len(indexed)
    scalenotes = [notes] * len(indexed)
    if len(keyframes) > 0:
        all_piano_rolls.append(pr_i)
        all_pitch_duration.append(pd_i)
        all_indexed.append(indexed)
        all_keyframes.append(keyframes)
        all_measurenums.append(measurenums)
        all_filenames.append(filenames)
        all_keys.append(keys)
        all_modes.append(modes)
        all_scalenotes.append(scalenotes)

final = {}
final["piano_rolls"] = all_piano_rolls
final["pitch_duration"] = all_pitch_duration
final["indexed"] = all_indexed
final["keyframs"] = all_keyframes
final["measurenums"] = all_measurenums
final["filenames"] = all_filenames
final["keys"] = all_keys
final["modes"] = all_modes
final["scalenotes"] = all_scalenotes
np.savez("music_data_1d.npz", **final)
print("Dumped results to 'music_data_1d.npz'")
