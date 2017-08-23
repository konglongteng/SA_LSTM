from scipy.misc import imread,imresize
import numpy as np
import random, json, time, sys

from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import img_to_array, load_img

def load_array_img(path, target_size=(224, 224)) :
    return img_to_array(load_img(path, target_size=target_size))

def train_generator1(batchsize, data_dir) :
    """
    read data stored in feature packages style.
    """
    if (data_dir[-1]!="/") :
        data_dir+= "/"
    meta = json.load(open(data_dir+"meta.json"))
    packages = []
    for length, nb_packs in meta :
        for pack in range(nb_packs) :
            packages.append((length, pack))
    random.shuffle(packages)

    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        perm = np.random.permutation(len(a))
        return a[perm], b[perm]

    while True :
        for (l, p) in packages :
            X = np.load(data_dir+"%02i-%02i-features.npy" % (l, p))
            Y = np.load(data_dir+"%02i-%02i-sentences.npy" % (l, p))
            X, Y = unison_shuffled_copies(X, Y)
            XL = np.ones((X.shape[0], 1)) * l
            for i in range(0, X.shape[0], batchsize) :
                yield [X[i:i+batchsize], XL[i:i+batchsize]], Y[i:i+batchsize]

def train_generator2(batchsize, features_dir, words_file, sentences_file, preload=15) :
    """
    read data stored in raw package style.
    """
    if (features_dir[-1]!="/") :
        features_dir+= "/"

    raw_sentences = json.load(open(sentences_file))
    word_map = dict([(str(x[1]),x[0]) for x in enumerate(np.load(words_file))])
    number_sentences = len(raw_sentences)
    file_buckets = {}
    sent_buckets = {}
    for file, raw_sent in raw_sentences :
        np_sent = np.array([[word_map[w]] for w in raw_sent.split()], dtype="int32")
        length = len(np_sent)
        if length in file_buckets :
            file_buckets[len(np_sent)].append(str(file)+".npy")
            sent_buckets[len(np_sent)].append(np_sent)
        else :
            file_buckets[len(np_sent)] = [str(file)+".npy"]
            sent_buckets[len(np_sent)] = [np_sent]
    available_lengths = list(file_buckets.keys())
    probabilities = [1.*len(file_buckets[x])/number_sentences for x in available_lengths]
    while True :
        chosen_length = np.random.choice(available_lengths, p=probabilities)
        indices = random.sample(xrange(len(file_buckets[chosen_length])), preload*batchsize)
        chosen_files = [file_buckets[chosen_length][i] for i in indices]
        
        X = [np.load(features_dir + f).reshape((5*5, 2048)) for f in chosen_files]
        X = np.array(X, dtype="float32")
        Y = [sent_buckets[chosen_length][i] for i in indices]
        Y = np.array(Y, dtype="int32")
        for i in range(0, batchsize*preload, batchsize) :
            yield [X[i:i+batchsize], Y[i:i+batchsize, :, 0]], Y[i:i+batchsize]

def train_generator_vgg16(batchsize, data_dir, words_file, sentences_file, phase="train") :
    """
    read data stored in raw style - seperate images.
    """
    if (data_dir[-1]!="/") :
        data_dir+= "/"

    raw_sentences = json.load(open(sentences_file))
    word_map = dict([(str(x[1]),x[0]) for x in enumerate(np.load(words_file))])
    number_sentences = len(raw_sentences)
    file_buckets = {}
    sent_buckets = {}
    for file, raw_sent in raw_sentences :
        np_sent = np.array([[word_map[w]] for w in raw_sent.split()], dtype="int32")
        length = len(np_sent)
        if length in file_buckets :
            file_buckets[len(np_sent)].append("COCO_%s2014_%012i.jpg"%(phase, file))
            sent_buckets[len(np_sent)].append(np_sent)
        else :
            file_buckets[len(np_sent)] = ["COCO_%s2014_%012i.jpg"%(phase, file)]
            sent_buckets[len(np_sent)] = [np_sent]
    available_lengths = list(file_buckets.keys())
    probabilities = [1.*len(file_buckets[x])/number_sentences for x in available_lengths]

    while True :
        chosen_length = np.random.choice(available_lengths, p=probabilities)
        indices = random.sample(xrange(len(file_buckets[chosen_length])), batchsize)
        chosen_files = [file_buckets[chosen_length][i] for i in indices]
        X = np.zeros((batchsize, 224, 224, 3), dtype="float32")
        Y = np.zeros((batchsize, chosen_length, 1), dtype="int32")
        for i, f in enumerate(chosen_files) :
            img = load_array_img(data_dir+f)
            if (np.random.rand()<0.5) :
                X[i] = np.fliplr(img)
            else :
                X[i] = img
            Y[i] = sent_buckets[chosen_length][indices[i]]
        X = preprocess_input(X)
        yield [X, Y[:, :, 0]], Y

def val_generator(batchsize, data_dir, features_dir, sentences_file, phase="val") :
    """
    return a batch of images and lists of their captions.
    """
    if (data_dir[-1]!="/") :
        data_dir+= "/"
    if (features_dir[-1]!="/") :
        features_dir+= "/"

    file_sentence = json.load(open(sentences_file))
    file_sentence = [("COCO_%s2014_%012i.jpg"%(phase, x[0]), x[1]) for x in file_sentence]
    file_map = {}
    for file, sent in file_sentence :
        if file in file_map :
            file_map[file].append(sent)
        else :
            file_map[file] = [sent]
    files = file_map.keys()
    while True :
        chosen_files = random.sample(files, batchsize)
        images = []
        captions = []
        features = []
        for file in chosen_files :
            images.append(load_array_img(data_dir+file))
            if (file in file_map) :
                captions.append(file_map[file])
            else :
                captions.append([])
            feature_file = str(int(file[15:-4]))+".npy"
            features.append(np.load(features_dir+feature_file).reshape(5*5, 2048))
        yield images, captions, features

def val_generator_vgg16(batchsize, data_dir, sentences_file, phase="val") :
    """
    return a batch of images and lists of their captions.
    """
    if (data_dir[-1]!="/") :
        data_dir+= "/"

    file_sentence = json.load(open(sentences_file))
    file_sentence = [("COCO_%s2014_%012i.jpg"%(phase, x[0]), x[1]) for x in file_sentence]
    file_map = {}
    for file, sent in file_sentence :
        if file in file_map :
            file_map[file].append(sent)
        else :
            file_map[file] = [sent]
    files = file_map.keys()
    while True :
        chosen_files = random.sample(files, batchsize)
        images = []
        captions = []
        X = np.zeros((batchsize, 224, 224, 3), dtype="float32")
        for i, file in enumerate(chosen_files) :
            images.append(load_array_img(data_dir+file))
            X[i] = images[-1].copy()
            if (file in file_map) :
                captions.append(file_map[file])
            else :
                captions.append([])
        X = preprocess_input(X)
        yield images, captions, X

# Test
if __name__ == "__main__" :
    VAL_FEATURES_DIR    = "/media/pqhuy98/222658B2265888A5/Datasets/MSCOCO/pretrained-vectors/inception-v3/val/"
    VAL_SENTENCES_FILE  = "val_sentences.json"
    VAL_DATA_DIR        = "MSCOCO/val2014/"

    TRAIN_DATA_DIR        = "MSCOCO/train2014/"
    TRAIN_SENTENCES_FILE  = "train_sentences.json"
    TRAIN_WORDS_FILE      = "embedding/words.npy"

    
    # g = val_generator_vgg16(32, VAL_DATA_DIR, VAL_SENTENCES_FILE)
    # while True :
    #     s = time.time()
    #     img, caps, X = next(g)
    #     print time.time()-s

    g = train_generator_vgg16(32,
        data_dir =       TRAIN_DATA_DIR,
        sentences_file = TRAIN_SENTENCES_FILE,
        words_file =     TRAIN_WORDS_FILE)
    while True :
        s = time.time()
        [X, YP], Y = next(g)
        print time.time()-s
        # for i in range(len(images)) :
        #     for cap in captions[i] :
        #         print cap
        #     print
        #     plt.figure()
        #     plt.imshow(images[i])
        #     plt.show()