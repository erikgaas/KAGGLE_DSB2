import glob
from PIL import Image
import numpy as np
pic_path = '/scratch/gaas0012/calc/center_find/*'

all_pics = glob.glob(pic_path)

ls = []

for ind, i in enumerate(all_pics):
	if ind % 1000 == 0:
		print(ind)
	im = np.asarray(Image.open(i))
	ls.append(np.reshape(im, (120,120,1)))


result = np.stack(ls)

np.save('/scratch/gaas0012/calc/all_pics.npy', result)