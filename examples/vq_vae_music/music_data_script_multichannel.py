import copy
import numpy as np
from collections import OrderedDict

from tfbldr.datasets import notes_to_midi
from tfbldr.datasets import midi_to_notes
from tfbldr.datasets import save_image_array

d = np.load("music_data_jos_1d.npz")
all_notes = copy.deepcopy(d['functional_notes'])
all_scalenotes = copy.deepcopy(d['scalenotes'])
all_chordnames = copy.deepcopy(d['chords_names'])
# scale tones in 5 octaves + 1 for rest
#v_imsize = 7 * 5 + 1
v_imsize = 48
h_imsize = 48
oh_lu = np.eye(v_imsize).astype("uint8")
all_measures_as_images = []
all_scalenotes_save = []
all_chordnames_save = []
all_midi_to_norm_kv = []
all_note_to_norm_kv = []
for iii in range(len(all_notes)):
    print(iii)
    scalenotes = all_scalenotes[iii][0] # scale constant over piece
    note_to_norm_lu = OrderedDict()
    #note_to_norm_lu["R"] = 0
    midi_to_norm_lu = OrderedDict()
    #midi_to_norm_lu[0] = 0
    # find lowest and highest notes , image
    min_s_m = np.inf
    min_s = None
    max_s_m = -np.inf
    max_s = None
    for s in scalenotes:
        m1 = notes_to_midi([[s + "1"]])[0][0]
        m5 = notes_to_midi([[s + "5"]])[0][0]
        if m1 < min_s_m:
            min_s_m = m1
            min_s = s
        if m5 > max_s_m:
            max_s_m = m5
            max_s = s

    reordered_scale = []
    start = False
    for s in scalenotes + scalenotes:
        if s == min_s:
            start = True
        if start:
            reordered_scale.append(s)
        if start and s == max_s:
            break

    # start from 1 if keeping rest channel
    #counter = 1
    counter = 0
    for octave in ["1", "2", "3", "4", "5"]:
        for note in reordered_scale:
            nnm = notes_to_midi([[note + octave]])[0][0]
            nn = midi_to_notes([[nnm]])[0][0][:-1]
            note_to_norm_lu[note + octave] = counter
            midi_to_norm_lu[nnm] = counter
            counter += 1

    measures_as_images = []
    for mi in range(len(all_notes[iii])):
        measure = all_notes[iii][mi]
        midi_m = notes_to_midi(measure)
        im = np.zeros((v_imsize, h_imsize, 4)).astype("uint8")
        for v in range(len(midi_m[0])):
            for t in range(len(midi_m)):
                # skip rest items, we will infer them in decoding
                if midi_m[t][v] != 0:
                    im[:, t, v] = oh_lu[midi_to_norm_lu[midi_m[t][v]]]

        im = im.astype("float32")
        measures_as_images.append(im)
        # track in case something is skipped
    all_measures_as_images.append(measures_as_images)
    all_scalenotes_save.append([scalenotes] * len(measures_as_images))
    all_chordnames_save.append(all_chordnames[iii])
    all_midi_to_norm_kv.append([(k, v) for k, v in midi_to_norm_lu.items()])
    all_note_to_norm_kv.append([(k, v) for k, v in note_to_norm_lu.items()])

final = {}
final["measures_as_images"] = all_measures_as_images
final["scalenotes"] = all_scalenotes_save
final["chordnames"] = all_chordnames_save
final["midi_to_norm_kv"] = all_midi_to_norm_kv
final["note_to_norm_kv"] = all_note_to_norm_kv
print("Dumping to music_data_multichannel.npz")
np.savez("music_data_multichannel.npz", **final)
