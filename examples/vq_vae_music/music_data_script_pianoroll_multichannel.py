import copy
import numpy as np
from collections import OrderedDict

from tfbldr.datasets import notes_to_midi
from tfbldr.datasets import midi_to_notes
from tfbldr.datasets import save_image_array
from tfbldr.datasets import piano_roll_imlike_to_image_array

d = np.load("music_data_jos_1d.npz")
all_notes = copy.deepcopy(d['functional_notes'])
all_scalenotes = copy.deepcopy(d['scalenotes'])
all_chordnames = copy.deepcopy(d['chords_names'])
# 88 + extra for nice multiplier
v_imsize = 96
h_imsize = 48
oh_lu = np.eye(v_imsize).astype("uint8")
all_measures_as_images = []
all_scalenotes_save = []
all_chordnames_save = []
for iii in range(len(all_notes)):
    print(iii)
    measures_as_images = []
    scalenotes = all_scalenotes[iii]
    for mi in range(len(all_notes[iii])):
        measure = all_notes[iii][mi]
        midi_m = notes_to_midi(measure)
        im = np.zeros((v_imsize, h_imsize, 4)).astype("uint8")
        for v in range(len(midi_m[0])):
            for t in range(len(midi_m)):
                # skip rest items, we will infer them in decoding
                if midi_m[t][v] != 0:
                    im[:, t, v] = oh_lu[midi_m[t][v]]

        im = im.astype("float32")
        measures_as_images.append(im)
        # track in case something is skipped
    all_measures_as_images.append(measures_as_images)
    all_scalenotes_save.append([scalenotes] * len(measures_as_images))
    all_chordnames_save.append(all_chordnames[iii])

"""
cc = np.concatenate([am[None] for am in all_measures_as_images[0][:16]], axis=0)
rr = piano_roll_imlike_to_image_array(cc, 0.25)
save_image_array(rr, "tmppr.png")
"""

final = {}
final["measures_as_images"] = all_measures_as_images
final["scalenotes"] = all_scalenotes_save
final["chordnames"] = all_chordnames_save
print("Dumping to music_data_pianoroll_multichannel.npz")
np.savez("music_data_pianoroll_multichannel.npz", **final)
