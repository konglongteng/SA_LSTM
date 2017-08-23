from nltk.translate.bleu_score import sentence_bleu as bleu
from scipy.misc import imsave
import numpy as np
import keras, json

class SaveSample(keras.callbacks.Callback):
    def __init__(self, output_dir, words_file, generator, max_lengths=18) :
        super(SaveSample, self).__init__()
        if (output_dir[-1]!="/") :
            output_dir+= "/"
        self.output_dir = output_dir
        self.i2w = dict(enumerate(np.load(open(words_file))))
        self.generator = generator
        self.max_lengths = max_lengths
        self.YL = None

    def on_epoch_end(self, epoch, logs={}):
        images, captions, X = next(self.generator)
        if self.YL is None :
            self.YL = np.ones((len(images), self.max_lengths), dtype="int32")
        X = np.array(X, dtype="float32")
        # Y_pred = np.random.uniform(size=(len(X), self.max_lengths, 5000))
        Y_pred = self.model.predict([X, self.YL])
        Y_pred = np.argmax(Y_pred, axis=-1)
        with open(self.output_dir+"index.html", "w") as f :
            f.write("<h1> Epoch "+str(epoch+1)+"</h1>")
            f.write("<h3> Trained : "+str((epoch+1)*1000*32)+" examples.</h3>")
            f.write("<h3> Loss : "+str(logs.get("loss"))+"</h3>")
            f.write("<h3> acc : "+str(logs.get("acc"))+"</h3><hr>")
            for i in range(len(images)) :
                s = " ".join([self.i2w[int(k)] for k in Y_pred[i]])
                s = s.split(" .")[0].strip()+" ."
                path = self.output_dir+("images/%i.jpg" % i)
                imsave(path, images[i])
                f.write(
                """
                <div>
                <img src='%s' style="width:255px;height:255px;"><br><br>
                <b>Output :</b> %s<br><br>
                <b>Expected :</b>
                <ul style="margin-top : 0">""" % ("images/%i.jpg"%i, s))
                for cap in captions[i] :
                    f.write("""
                    <li>%s</li>""" % cap)
                f.write("""
                </ul>
                <b>BLEU score :</b> %.3f<br>
                <hr>
                </div>
                """ % bleu(captions[i], s))

# Test
if __name__ == "__main__" :
    import data_generator
    g =  data_generator.val_generator(32,
        "/media/pqhuy98/222658B2265888A5/Datasets/MSCOCO/val2014/",
        "/media/pqhuy98/222658B2265888A5/Datasets/MSCOCO/pretrained-vectors/inception-v3/val/",
        "val_sentences.json")
    ss = SaveSample("sample/", "embedding/words.npy", g)
    ss.on_epoch_end(1)