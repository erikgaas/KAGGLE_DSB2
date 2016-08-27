import glob
import Image
import numpy as np
pic_path = '/scratch/gaas0012/calc/center_find/*'

all_pics = glob.glob(pic_path)

ls = []

for ind, i in enumerate(all_pics):
	if ind % 1000 == 0:
		print(ind)
	im = Image.open(i)
	pixels = im.load()
	ls.append(pixels)


result = np.vstack(ls)

np.save('/scratch/gaas0012/calc/all_pics.npy', result)