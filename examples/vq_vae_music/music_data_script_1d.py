import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy

import os
import numpy as np
from collections import Counter, OrderedDict
from tfbldr.datasets import notes_to_midi
from tfbldr.datasets import midi_to_notes
from functools import reduce

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
all_functional_notes = []
all_functional_notes_idx = []
all_functional_notes_kv = []
all_functional_voicings_idx = []
all_functional_voicings_kv = []
all_keyframes = []
all_chords_names = []
all_indexed = []
all_absolutes = []
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
    loaded_chords_names = copy.deepcopy(d["chords_names"])
    key = d["keyname"]
    mode = d["keymode"]
    # last note is octave of the root, skip it
    notes = d["keynotes"][:-1]
    assert sorted(list(set(d["keynotes"]))) == sorted(list(notes))

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

    piano_rolls = []
    pitch_duration = []
    measurenums = []
    keyframes = []
    indexed = []
    absolutes = []
    keys = []
    modes = []
    chords_names = []
    scalenotes = []
    functional_notes = []
    functional_notes_idx = []
    functional_voicings_idx = []

    func_notes_lu = {}
    func_notes_lu["R"] = 0
    # R is always in the lowest voicing -> R0
    counter = 1
    notes = [n for n in notes]
    for n1 in notes + ["R"]:
        for n2 in notes + ["R"]:
            for n3 in notes + ["R"]:
                for n4 in notes + ["R"]:
                     # hack to represent it in the form we get from midi_to_notes
                     # basically changing E-3 -> Eb3 , etc
                     if n1 != "R":
                         nnn1 = midi_to_notes(notes_to_midi([[n1 + octave]]))[0][0][:-1]
                     else:
                         nnn1 = n1

                     if n2 != "R":
                         nnn2 = midi_to_notes(notes_to_midi([[n2 + octave]]))[0][0][:-1]
                     else:
                         nnn2 = n2

                     if n3 != "R":
                         nnn3 = midi_to_notes(notes_to_midi([[n3 + octave]]))[0][0][:-1]
                     else:
                         nnn3 = n3

                     if n4 != "R":
                         nnn4 = midi_to_notes(notes_to_midi([[n4 + octave]]))[0][0][:-1]
                     else:
                         nnn4 = n4
                     func_notes_lu[tuple([nnn1, nnn2, nnn3, nnn4])] = counter
                     counter += 1

    func_voicings_lu = {}
    # hardcode for 4 voices for now
    count = 0
    for o1 in [0, 1, 2, 3, 4, 5]:
        for o2 in [0, 1, 2, 3, 4, 5]:
            for o3 in [0, 1, 2, 3, 4, 5]:
                for o4 in [0, 1, 2, 3, 4, 5]:
                    oo = [o1, o2, o3, o4]
                    nz = [ooi for ooi in oo if ooi != 0]
                    # can only be an ordering with at least 2
                    if len(nz) == 0 or len(nz) == 1:
                        func_voicings_lu[tuple(oo)] = count
                        count += 1
                    else:
                        rr = range(len(nz))
                        ordered = True
                        maxv = 5
                        for i in rr:
                            if nz[i] <= maxv:
                                maxv = nz[i]
                            else:
                                ordered = False
                        # allow voice crossing in the middle 2 voices?
                        if ordered:
                            func_voicings_lu[tuple(oo)] = count
                            count += 1

    inv_func_voicings = {v: k for k, v in func_voicings_lu.items()}
    last_non_rest = [0, 0, 0, 0]
    for n in range(len(prs)):
        # key and mode delta normalized repr
        # 0 rest, 1:28 is [-13, 14]
        pr_i = prs[n]
        pd_i = pds[n]
        chords_names_i = loaded_chords_names[n]
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

        func = midi_to_notes(pr_i)
        func_notes = copy.deepcopy(func)
        func_notes_i = np.zeros_like(pr_i[:, 0])
        func_voicings_i = np.zeros_like(pr_i[:, 0])
        """
        loop_reset = False
        for iii in range(len(pr_i)):
            fvi = [int(fi[-1]) if fi != "R" else 0 for fi in func[iii]]
            if tuple(fvi) not in func_voicings_lu:
                print("unknown voicing {}".format(fvi))
                loop_reset = True
                break
            fni = [fi[:-1] if fi != "R" else "R" for fi in func[iii]]
            fni_idx = func_notes_lu[tuple(fni)]
            fvi_idx = func_voicings_lu[tuple(fvi)]
            func_notes_i[iii] = fni_idx
            func_voicings_i[iii] = fvi_idx
        if loop_reset:
            continue
        """

        # put the scale notes in
        out = np.zeros_like(pr_i)
        for unote in np.unique(pr_i):
            out[pr_i == unote] = scale_lu[midi_lu[unote]]
        absolute = copy.deepcopy(out)

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
        chords_names.append(chords_names_i)
        measurenums.append(n)
        absolutes.append(absolute)
        piano_rolls.append(pr_i)
        pitch_duration.append(pd_i)
        functional_notes.append(func_notes)
        functional_notes_idx.append(func_notes_i)
        functional_voicings_idx.append(func_voicings_i)

        for v in range(pr_i.shape[-1]):
            non_rest = pr_i[pr_i[:, v] > 0, v]
            if len(non_rest) > 0:
                last_non_rest[v] = scale_lu[midi_lu[non_rest[-1]]]

    filenames = [fnpz] * len(indexed)
    keys = [key] * len(indexed)
    modes = [mode] * len(indexed)
    scalenotes = [notes] * len(indexed)
    if len(keyframes) > 0:
        all_piano_rolls.append(piano_rolls)
        all_pitch_duration.append(pitch_duration)
        all_indexed.append(indexed)
        all_keyframes.append(keyframes)
        all_chords_names.append(chords_names)
        all_absolutes.append(absolutes)
        all_functional_notes.append(functional_notes)
        all_functional_notes_idx.append(functional_notes_idx)
        all_functional_notes_kv.append([(k, v) for k, v in func_notes_lu.items()])
        all_functional_voicings_idx.append(functional_voicings_idx)
        all_functional_voicings_kv.append([(k, v) for k, v in func_voicings_lu.items()])
        all_measurenums.append(measurenums)
        all_filenames.append(filenames)
        all_keys.append(keys)
        all_modes.append(modes)
        all_scalenotes.append(scalenotes)

final = {}
final["piano_rolls"] = all_piano_rolls
final["pitch_duration"] = all_pitch_duration
final["indexed"] = all_indexed
final["absolutes"] = all_absolutes
final["keyframes"] = all_keyframes
final["chords_names"] = all_chords_names
final["functional_notes"] = all_functional_notes
final["functional_notes_idx"] = all_functional_notes_idx
final["functional_notes_kv"] = all_functional_notes_kv
final["functional_voicings_idx"] = all_functional_voicings_idx
final["functional_voicings_kv"] = all_functional_voicings_kv
final["measurenums"] = all_measurenums
final["filenames"] = all_filenames
final["keys"] = all_keys
final["modes"] = all_modes
final["scalenotes"] = all_scalenotes
np.savez("music_data_1d.npz", **final)
print("Dumped results to 'music_data_1d.npz'")
