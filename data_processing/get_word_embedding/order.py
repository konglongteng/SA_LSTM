from __future__ import division

import struct
import sys, json

used_words = set()
with open("train-1000.json") as f:
    data = json.load(f)
for _, s in data :
    ws = s.split()
    for w in ws :
        used_words.add(str(w))

FILE_NAME = "GoogleNews-vectors-negative300.bin"
MAX_VECTORS = 3000000 # This script takes a lot of RAM (>2GB for 200K vectors), if you want to use the full 3M embeddings then you probably need to insert the vectors into some kind of database
FLOAT_SIZE = 4 # 32bit float
words = []

with open(FILE_NAME, 'rb') as f:
    c = None
    # read the header
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
        words.append(word)
        binary_vector = f.read(FLOAT_SIZE * vector_len)

        sys.stdout.write("\rRead %i/%i." % (_, num_vectors))
        # sys.stdout.flush()

with open("unordered.json", "w") as f :
    json.dump(words, f)
words = sorted(words)
with open("ordered.json", "w") as f :
    json.dump(words, f)