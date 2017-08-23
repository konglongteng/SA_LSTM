from keras.layers import Input
from keras.layers.core import Reshape
from keras.models import Model
from keras.applications import *
from keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from keras.optimizers import RMSprop

from SA_LSTM import SA_LSTM
from callbacks import SaveSample
import data_generator

import tensorflow as tf
import numpy as np
import random, json
import time

TENSORBOARD_DIR = "tensorboard/resnet50/"
EMBEDDING_FILE = "embedding/matrix.npy"

TRAIN_FEATURES_DIR      = "train/"
TRAIN_SENTENCES_FILE    = "MSCOCO/train_sentences.json"
TRAIN_DATA_DIR        = "MSCOCO/train2014/"

VAL_FEATURES_DIR    = "/media/pqhuy98/222658B2265888A5/Datasets/MSCOCO/pretrained-vectors/inception-v3/val/"
VAL_SENTENCES_FILE  = "MSCOCO/val_sentences.json"
VAL_DATA_DIR        = "MSCOCO/val2014/"

WORDS_FILE        = "embedding/words.npy"
# VAL_FEATURES_DIR      = "train/"
# VAL_SENTENCES_FILE    = "train_sentences.json"
# VAL_DATA_DIR        = "/media/pqhuy98/222658B2265888A5/Datasets/MSCOCO/train2014/"

SAVE_DIR = "save/resnet50/"
HTML_DIR = "/var/www/html/ML/"
PRETRAIN = "resnet50"

locations = 5*5
features = 2048
vocabsize = 5000
units = 512
middle_units = 49

steps_per_epoch = 1000
fix_epochs = 20
epoch_per_layer = 10

def build(locations, features, vocabsize, units, middle_units, embedding_file, pretrain=None) :
    if pretrain is None :
        inp = x = Input((locations, features))
    elif pretrain == "resnet50" :
        cnn = resnet50.ResNet50(include_top=False, input_shape=(224, 224, 3))
        cnn.layers.pop()
        cnn.layers[-1].outbound_nodes = []
        cnn.outputs = [cnn.layers[-1].output]
        inp = cnn.inputs[0]
        x = cnn.outputs[0]
        x = Reshape((7*7, 2048))(x)
        nb_freeze = 172
    else :
        raise ValueError("pretrain is not understood : %s." % str(pretrain))

    prev_words = Input((None,))
    x = SA_LSTM(
        vocabsize=vocabsize,
        units=units,
        middle_units=middle_units,
        embeddings_weights= np.load(embedding_file),
        embeddings_trainable=False)([x, prev_words])
    model = Model([inp, prev_words], x)
    for layer in model.layers[:nb_freeze] :
        layer.trainable = False
    return model

if __name__ == "__main__" :
    model = build(
        locations, features, vocabsize,
        units, middle_units,
        EMBEDDING_FILE, pretrain=PRETRAIN)
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=RMSprop(lr=0.001),
        metrics=["acc"])
    model.summary()

    # model.load_weights("save/resnet50/weights-2.30903.hdf5")

    if (PRETRAIN is None) :
        train_g = data_generator.train_generator2(48,
            words_file =     FILE,
            features_dir =   TRAIN_FEATURES_DIR,
            sentences_file = TRAIN_SENTENCES_FILE)
        val_g = data_generator.val_generator(16,
            data_dir =       VAL_DATA_DIR,
            features_dir =   VAL_FEATURES_DIR,
            sentences_file = VAL_SENTENCES_FILE,
            phase="val")
    elif (PRETRAIN in ["resnet50", "vgg16"]) :
        train_g = data_generator.train_generator_vgg16(32,
            words_file =     WORDS_FILE,
            data_dir =       TRAIN_DATA_DIR,
            sentences_file = TRAIN_SENTENCES_FILE)
        val_g = data_generator.val_generator_vgg16(16,
            data_dir =       VAL_DATA_DIR,
            sentences_file = VAL_SENTENCES_FILE,
            phase="val")

    filepath = SAVE_DIR+"weights-{loss:.5f}.hdf5"
    cb1 = ModelCheckpoint(filepath, monitor='loss', save_best_only=True, mode='min')
    cb2 = SaveSample(HTML_DIR, WORDS_FILE, val_g)
    cb3 = TensorBoard(
        log_dir=TENSORBOARD_DIR,
        histogram_freq=1,
        write_graph=False,
        write_images=False,
        write_grads=False)

    cb = [cb1, cb2, cb3]
    initial_epoch = 0

    model.fit_generator(
        generator = train_g,
        steps_per_epoch = steps_per_epoch,
        epochs = fix_epochs,
        callbacks = cb,
        max_queue_size = 60,
        initial_epoch = initial_epoch)
    initial_epoch = fix_epochs

    freezed = []
    for layer in model.layers :
        if len(layer.non_trainable_weights) > 0 and not layer.trainable:
            freezed.append(layer)
    print "Freezed layers :",len(freezed)
    for x in freezed :
        print x.name

    lr = 0.0005
    for i in range(3) :
        lr*= 0.8
        print
        if i == 2 :
            nskip = 8
        else :
            nskip = 6
        for _ in range(min(len(freezed), nskip)) :
            layer = freezed.pop()
            layer.trainable = True
            print "Unfreeze", layer.name
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=RMSprop(lr),
            metrics=["acc"])
        count_freezed = 0
        for layer in model.layers :
            if len(layer.non_trainable_weights) > 0 and not layer.trainable:
                count_freezed+= 1
        print "Freezed layers :", count_freezed

        model.fit_generator(
            generator=train_g,
            steps_per_epoch=steps_per_epoch,
            epochs=initial_epoch+epoch_per_layer,
            callbacks=cb,
            max_queue_size=60,
            initial_epoch=initial_epoch)
        initial_epoch+= epoch_per_layer

    model.fit_generator(
        generator=train_g,
        steps_per_epoch=steps_per_epoch,
        epochs=initial_epoch+10000,
        callbacks=cb,
        max_queue_size=60,
        initial_epoch=initial_epoch)