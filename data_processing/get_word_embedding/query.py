import json

with open("unordered.json") as f :
	words = json.load(f)
words = set([x.lower() for x in words])

k = 5000
used_words = set()
with open("train-%i.json" % k) as f:
    data = json.load(f)
for _, s in data :
    ws = s.split()
    for w in ws :
        used_words.add(str(w))

print used_words - set(used_words).intersection(words)

# while True :
# 	w = raw_input("Query :")
# 	if (w in words) :
# 		print "Yes"
# 	else :
# 		print "No"