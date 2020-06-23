from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import load_model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from keras.optimizers import adam
from keras.layers.merge import add
from keras.models import Model


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Image Path")
args = vars(ap.parse_args())
img_path = args['image']

def load_pretrained_model(model_weights_dir,max_length,vocab_size):
        # features from the CNN model squeezed from 2048 to 256 nodes
        inputs1 = Input(shape=(2048,))
        fe1 = Dropout(0.5)(inputs1)
        fe2 = Dense(256, activation='relu')(fe1)
        # LSTM sequence model
        inputs2 = Input(shape=(max_length,))
        se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
        se2 = Dropout(0.5)(se1)
        se3 = LSTM(256)(se2)
        # Merging both models
        decoder1 = add([fe2, se3])
        decoder2 = Dense(256, activation='relu')(decoder1)
        outputs = Dense(vocab_size, activation='softmax')(decoder2)
        # tie it together [image, seq] [word]
        model = Model(inputs=[inputs1, inputs2], outputs=outputs)
        opt=adam(learning_rate=1e-3)
        model.load_weights(model_weights_dir)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model
        

def extract_features(filename, model):
        try:
            image = Image.open(filename)
            
        except:
            print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature

def word_for_id(integer, tokenizer):
 for word, index in tokenizer.word_index.items():
     if index == integer:
         return word
 return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = '<Start>'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text





max_length = 76
vocab_size=19728
tokenizer = load(open("Tokenizer.p","rb"))
model = load_pretrained_model('pretrained_weights.h5',max_length,vocab_size)
xception_model = Xception(include_top=False, pooling="avg")

img = Image.open(img_path)
photo = extract_features(img_path, xception_model)

generated_caption=generate_desc(model,tokenizer,photo,max_length)
caption = generated_caption.split()[1].capitalize() 
for x in generated_caption.split()[2:len(generated_caption.split())-1]:
	caption = caption + ' ' + x
caption += '.'
print(caption,"\n\n\n")
plt.imshow(img)
