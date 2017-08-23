import json
import numpy as np
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import random, sys

k = 5000

data = json.load(open("train-%i.json" % k))
words = np.load("processed/words.npy")
w2i = {}
for i, w in enumerate(words) :
	w2i[w] = i

X = {}
Y = {}

for x, s in data :
	ws = s.split()
	l = len(ws)
	if l in X :
		X[l].append(x)
	else :
		X[l] = [x]
	y = [w2i[w] for w in ws]
	if l in Y :
		Y[l].append(y)
	else :
		Y[l] = [y]

save_dir = "/media/pqhuy98/142AABD82AABB560/FastStorage/packs/inception-v3/"
pack_size = 1000
meta = {}

start_l = 11
start_i = 68000

for l in X :
	print "Length", l
	ids = X[l]
	size = len(ids)
	meta[l] = max(1,size/pack_size)
    if l<start_l :
        continue
	sents = list(np.expand_dims(np.array(Y[l], dtype="int32"), axis=-1))
	pairs = zip(ids, sents)
	random.shuffle(pairs)
	for i in range(0, size, pack_size) :
        if (l == start_l and i<start_i) :
            continue
		x, y = [], []
		if size-i<pack_size*2 :
			bonus = pack_size
		else :
			bonus = 0
		for id, sent in pairs[i:i+pack_size+bonus] :
			fv = np.load("../../Datasets/MSCOCO/pretrained-vectors/inception-v3/train/"+str(id)+".npy")
			fv = np.reshape(fv, (5*5, 2048))
			x.append(fv)
			y.append(sent)
		x = np.array(x)
		y = np.array(y)
		print x.shape, y.shape
		np.save(save_dir+"%02i-%02i-features.npy"%(l, i/pack_size), x)
		np.save(save_dir+"%02i-%02i-sentences.npy"%(l, i/pack_size), y)
		print "\r%i/%i" % (min(size,i+pack_size+bonus), size),
		sys.stdout.flush()
		del x
		del y
		if bonus :
			break
	print

json.dump(meta.items(), open(save_dir+"meta.json", "w"))