"""
Create cleaned json versions contain only 1000, 2000, 3000... most frequent words.
Calculate word prequency.
Should be used for training data only.
"""

import json, re
import random

# Keep top_k most frequent words
# top_k = 500


for phase in ["train"] :#, "val"] :
    words = []
    data = json.load(open("raw/semi-clean-"+phase+".json"))
    print phase, len(data)
    for x in data :
        ws = x[1].split()
        words+= ws

    freq = {}
    for w in words :
        if (w in freq) :
            freq[w]+= 1
        else :
            freq[w] = 1

    words = list(freq.items())
    words = sorted(words, key=lambda x : x[1], reverse=True)

    for top_k in range(1000, 5001, 1000) :
        # For some values of k, some chosen words will not appear in any chosen sentences.
        # If you don't understand why, think more about it.
        choosen_words = [x[0] for x in words[:top_k]]
        choosen_words = set(choosen_words)
        cleaned = []
        for x in data :
            all_in = all([w in choosen_words for w in x[1].split()])
            if (all_in) :
                cleaned.append(x)
        print "Top", top_k, "words - Number of samples :", len(cleaned)
        with open("cleaned/"+phase+"-"+str(top_k)+".json", "w") as f :
            json.dump(cleaned, f)
        used_words = set()
        for _, s in cleaned :
            ws = s.split()
            for w in ws :
                used_words.add(str(w))
        print choosen_words-used_words

with open("words-frequency.txt", "w") as f :
    for x in words :
        f.write("%-15s%i\n" % (x[0],x[1]))