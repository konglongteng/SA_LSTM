"""
Remove irrelevant json data.
Remove images which have extremely skewed size, ie 100*600, 600*64...
Clean words.
Remove very long sentences.
Remove sentences which contain some special characters (ie. ", ?, ').
"""

import json, re

def clean(s) :
    res = s.lower()
    while (res[-1] in [".", ","]) :
        res = res[:-1]
    res = " "+res+" "
    res = re.sub(r"\. ", ", ", res)
    res = re.sub(r"\.", "", res)
    res = re.sub(r"\,"," , ", res)
    res = re.sub(r"-", " ", res)
    res = re.sub(r" can't ", " cannot ", res)
    res = re.sub(r" skiies ",   " skies ", res)
    res = re.sub(r"frisbe ",    "frisbee ", res)
    res = re.sub(r"frissbee ",  "frisbee ", res)
    res = re.sub(r"frisbey ",   "frisbee ", res)
    res = re.sub(r"frisee ",    "frisbee ", res)
    res = re.sub(r"frizbe ",    "frisbee ", res)
    res = re.sub(r"frizbee ",   "frisbee ", res)
    res = re.sub(r"frizbees ",  "frisbees ", res)
    res = re.sub(r" deckered ", " decker ", res)
    res = re.sub(r" frisky ",   " playful ", res)
    res = re.sub(r" rared ",    " rare ", res)
    res = re.sub(r" selfie ",   " portrait ", res)
    res = re.sub(r" its' ",     " it's ",  res)
    res = re.sub(r"n't ",       " not ", res)
    res = re.sub(r"\'re ",      " are ", res)
    res = re.sub(r"\'ve ",       " have ", res)
    res = re.sub(r"\'d ",        " would ", res)
    res = re.sub(r"\'ll ",       " will ", res)
    res = re.sub(r"\'s", " 's ", res)
    res = " ".join(str(res.strip()).split())
    res+= " ."
    return res

# Ratio height/width within range [lo; hi]
lo, hi = {}, {}
lo["train"] = 0.0001
lo["val"] = 0.0001
hi["train"] = 1./lo["train"]
hi["val"] = 1./lo["val"]

max_len = 18
min_len = 8

for phase in ["train", "val"] :
    with open("raw/"+phase+".json") as f:
        data = json.load(f)
    #--
    img = data["images"]
    nb_images = 0
    path = {}
    for x in img :
        if (lo[phase] <= 1.*x["height"]/x["width"] <= hi[phase]) :
            path[x["id"]] = x["file_name"]
            nb_images+= 1
    #--
    cap = data["annotations"]
    ic = []
    for x in cap :
        s = clean(x["caption"])
        nb_words = len(s.split())
        if (x["image_id"] in path) and min_len <= nb_words <= max_len :
            if re.match("[a-z\.,'\- ]+$", s) :
                ic.append((x["image_id"], s))
    ic = list(set(ic))
    with open("raw/semi-clean-"+phase+".json", "w") as f :
        json.dump(ic, f)
    print phase
    print "Number of images :", nb_images
    print "Number of sentences :", len(ic)