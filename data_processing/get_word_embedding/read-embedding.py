"""
Geneterate word embedding matrix for top 5000 words.
Weight values are from word2vec.
"""

from __future__ import division

from scipy.spatial.distance import cosine
from random import gauss
import numpy as np
import struct
import sys, json

# for k in range(0, 10001, 1000) :
k = 5000
used_words = set()
with open("train-%i.json" % k) as f:
    data = json.load(f)
for _, s in data :
    ws = s.split()
    for w in ws :
        used_words.add(str(w))
print len(used_words)

emb = []
words = []
found = set()

FILE_NAME = "GoogleNews-vectors-negative300.bin"
MAX_VECTORS = 3000000 # This script takes a lot of RAM (>2GB for 200K vectors), if you want to use the full 3M embeddings then you probably need to insert the vectors into some kind of database
FLOAT_SIZE = 4 # 32bit float

with open(FILE_NAME, 'rb') as f:
    # read the header
    c = None
    header = ""
    while c != "\n":
        c = f.read(1)
        header += c
    total_num_vectors, vector_len = (int(x) for x in header.split())
    num_vectors = min(MAX_VECTORS, total_num_vectors)
    print "Number of vectors: %d/%d" % (num_vectors, total_num_vectors)
    print "Vector size: %d" % vector_len
    for _ in range(num_vectors):
        word = ""        
        while True:
            c = f.read(1)
            if c == " ":
                break
            word += c
        word = word.lower()
        binary_vector = f.read(FLOAT_SIZE * vector_len)
        vector = [ struct.unpack_from('f', binary_vector, i)[0] 
                          for i in xrange(0, len(binary_vector), FLOAT_SIZE) ]
        if (word in used_words) and not (word in found) :
            found.add(word)
            words.append(word)
            emb.append(np.array(vector, dtype="float32"))
            if (len(found)+2 == len(used_words)) :
                break

        sys.stdout.write("\rRead %i/%i, found %i/%i" % (_, num_vectors, len(found), len(used_words)))
        sys.stdout.flush()
missing = list(used_words - found)

data = dict(zip(words, emb))

def closed_distance(vector) :
    d = [cosine(vector, data[x]) for x in data]
    return min(d)

def random_unit_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return np.array([x/mag for x in vec])

norm = [np.linalg.norm(data[x]) for x in data]
norm = min(norm), max(norm)
print "Norm range :", norm
for c in missing :
    print "-"*20
    print "Search for", c
    res = data["a"]
    max_dis = 0
    for i in range(100) :
        tmp = random_unit_vector(res.shape[0]) * (norm[0]+np.random.rand()*(norm[1]-norm[0]))
        dis = closed_distance(tmp)
        if (dis > max_dis) :
            res = tmp
            max_dis = dis
    print "Closed dis :", closed_distance(res)
    print "Norm tmp :", np.linalg.norm(res)
    data[c] = res

print len(list(data.keys()))

data = sorted(list(data.items()), key=lambda x:x[0])

words = [str(x[0]) for x in data]
matrix = np.array([x[1] for x in data], dtype="float32")

print matrix.shape, matrix.dtype

np.save("processed/words.npy", words)
np.save("processed/matrix.npy", matrix)