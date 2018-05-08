import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
import numpy as np
from collections import Counter
from skimage.transform import resize

basedir = "/u/kastner/music_npz"
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
# truncate to +- 2 octaves
measure_count = 0
to_keep_measures = []
to_keep_centers = [] 
to_keep_keys = []
to_keep_modes = []
to_keep_scale_notes = []
for fnpz in sorted(os.listdir(basedir)):
    print(fnpz)
    try:
        d = np.load(basedir + os.sep + fnpz)
    except:
        print("Unable to load {}, continuing".format(fnpz))
    if len(d['centered']) < 1 or 'keyname' not in d:
        print(fnpz + " had zero length or no key")
        continue
    for mi in range(len(d['centered'])):
        measure = d['centered'][mi]
        # this drops any measure which introduces or removes rest - not ideal for music... but acceptable for now
        if (measure[measure > -99] > 24).any() or (measure[measure > -99] < -23).any():
            pass
        else:
            measure_count += 1
            to_keep_measures.append(d['centered'][mi])
            to_keep_centers.append(d['offset'][mi])
            to_keep_keys.append(d['keyname'])
            to_keep_modes.append(d['keymode'])
            to_keep_scale_notes.append(d['keynotes'])


mapper = {m: n for n, m in enumerate(np.arange(-23, 24 + 1))}

measures_as_images = []
measures_indices = []
for n in range(len(to_keep_measures)):
    if not (n % 1000):
        print("processing {}".format(n))
    if to_keep_measures[n].shape[-1] == 4:
        this_measure = to_keep_measures[n][:, :-1]
    elif to_keep_measures[n].shape[-1] == 3:
        this_measure = to_keep_measures[n]
    else:
        print("measure {} size not 4 or 3, skip it".format(n))
        continue

    if (this_measure > -99).sum() == 0:
        print("measure {} all rest, skip it".format(n))
        continue
    v_imsize = 48 # (-23 to +24, inclusive w 0 = 48) 
    # gonna have to resize or something...
    h_imsize = len(this_measure)
    # 3 voices, ordered 0 == bass, 1 == tenor, 2 == soprano/alto
    # mask ordering will be bass (least conditioning) -> soprano (most conditioning)
    im = np.zeros((v_imsize, h_imsize, 3)).astype("float32")
    for c in list(range(this_measure.shape[-1])):
        for ii in np.arange(-23, 24 + 1):
            # 3 voices, ordered 0 == bass, 1 == tenor, 2 == soprano/alto, happens here
            im[mapper[ii], :, 2 - c] = 1. * (this_measure[:, c] == ii)

    measures_as_images.append(im)
    # track in case something is skipped
    measures_indices.append(n)
    """
    if (this_measure[:, 0] > -99).sum() != 0 and (this_measure[:, 1] > -99).sum() != 0 and (this_measure[:, 2] > -99).sum() != 0:
        print("n, {}".format(n))
        plt.imshow(im, origin="lower")
        plt.savefig("exim.png")
    """

final = {}
final["measures"] = []
final["centers"] = []
final["keys"] = []
final["modes"] = []
final["scale_notes"] = []
for i in range(len(measures_as_images)):
    ti = measures_as_images[i]
    li = measures_indices[i]
    if ti.shape != (48, 48, 3):
        rsz = resize(ti, (48, 48, 3))
        rsz[rsz <= 0.5] = 0.
        rsz[rsz > 0.5] = 1.
        ti = rsz
    final["measures"].append(ti)
    final["centers"].append(to_keep_centers[li])
    final["keys"].append(to_keep_keys[li])
    final["modes"].append(to_keep_modes[li])
    final["scale_notes"].append(to_keep_scale_notes[li])
np.savez("music_data.npz", **final)
