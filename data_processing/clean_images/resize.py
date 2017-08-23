from scipy.misc import imread, imsave, imresize
import json, threading, time, sys
import traceback
import numpy as np

k = 1000

with open("caption/cleaned/train-"+str(k)+".json") as f :
	data = json.load(f)

print "Number of samples :", len(data)
id = [x[0] for x in data]
id = list(set(id))
print "Used images :", len(id)

cnt = 0
lock = threading.Lock()
event = threading.Event()

names = []
images = []

def resize(idx, nthread, id, event) :
	global cnt
	for i in range(idx,len(id),nthread) :
		if event.is_set() :
			break
		x = id[i]
		img = imread("train2014/"+x)
		img = imresize(img, (112, 112))
		if (len(img.shape)>2 and img.shape[2]>=3) :
			img = img[:,:,:3]
		else :
			if (len(img.shape) <= 2) :
				img = np.concatenate([np.expand_dims(img, axis=2)]*3, axis=2)
			else :
				img = np.concatenate([img[:,:,:1]]*3, axis=2)
		if (img.shape != (112, 112, 3)) :
			print img.shape
		with lock :
			names.append(x)
			images.append(img)
			cnt+= 1
			sys.stdout.write("\r%i/%i" % (cnt, len(id)))
			sys.stdout.flush()

nthread = 4
thr = []
for i in range(nthread) :
	thr.append(threading.Thread(target = resize, args = (i, nthread, id, event)))
for x in thr :
	x.start()
try :
	while cnt < len(id) :
		time.sleep(0.1)
	for x in thr :
		x.join()
	np.save("names.npy", names)
	np.save("images.npy", images)

except KeyboardInterrupt :
	event.set()
except :
	traceback.print_exc()