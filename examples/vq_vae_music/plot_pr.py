import numpy as np
from tfbldr.datasets import piano_roll_imlike_to_image_array
from tfbldr.datasets import save_image_array
import sys

fname = sys.argv[1]
d = np.load(fname)
xr = d["pr"][8:16]
rr = piano_roll_imlike_to_image_array(xr, 0.25, background="white")
rr = rr[:, 40:80]
pngname = "samples/{}.png".format(fname.split("/")[-1].split(".")[0])
print("saving {}".format(pngname))
save_image_array(rr, pngname, resize_multiplier=(4, 1), gamma_multiplier=7, flat_wide=True)
print("image complete")
from IPython import embed; embed(); raise ValueError()
