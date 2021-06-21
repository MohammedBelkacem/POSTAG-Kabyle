# Par Belkacem Mohammed
# Publié sous licence CC0
# Copyright Juillet 2021

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
import numpy as np
from pickle import load


def add_basic_features(sentence_terms, index):
    term = sentence_terms[index]
    return {
        'nb_terms': len(sentence_terms),
        'term': term,
        'is_first': index == 0,
        'is_last': index == len(sentence_terms) - 1,
        'is_capitalized': term[0].upper() == term[0],
        'is_all_caps': term.upper() == term,
        'is_all_lower': term.lower() == term,
        'is_numeric': sentence_terms[index].isdigit(),
        'prev_word': '' if index == 0 else sentence_terms[index - 1],
        'prev2_word': '' if index == 1 else sentence_terms[index - 2],
        'next_word': '' if index == len(sentence_terms) - 1 else sentence_terms[index + 1],
        'prefix-1': sentence_terms[index][0],
        'prefix-2': sentence_terms[index][:2],
        'prefix-3': sentence_terms[index][:3],
        'prefix-4': sentence_terms[index][:4],
        'prefix-5': sentence_terms[index][:5],
        'suffix-1': sentence_terms[index][-1],
        'suffix-2': sentence_terms[index][-2:],
        'suffix-3': sentence_terms[index][-3:],
        'suffix-4': sentence_terms[index][-4:],

    }

def transform_to_dataset(sentences):
    X = []

    for sentence in sentences:
        for index, word in enumerate(sentence) :
            X.append(add_basic_features(sentence, index))
    return X

sentences=[[('Awi'),('-d'),('aman'),('.')],[('Awi'),('-as'),('-t'),('-id'),('.'),('Zemren'),('i'),('twuɣa'),('qerriḥen'),('.')]]

X_train = transform_to_dataset(sentences)
# Fit our DictVectorizer with our set of features
dict_vectorizer = DictVectorizer(sparse=False)
dict_vectorizer.fit(X_train)

# Convert dict features to vectors
X_train = dict_vectorizer.transform(X_train)


# load model the kerras model
loaded_model = load(open('model.clf', 'rb'))
#load the input dim
for l in loaded_model.layers:
    inputs= l.input_shape[1]
    break
#add remaining dim for the sentences to be taged
a=np.zeros((X_train.shape[0], inputs-X_train.shape[1]))
X_train=np.concatenate((X_train, a),axis=1)
# make prediction
predictions = loaded_model.predict_classes(X_train,verbose=1)

#lod the tag labels
label_encoder = LabelEncoder()
label_encoder = load(open('label_encoder.pkl', 'rb'))

#print words and theur tags

words=[]
for i in sentences:
    for j in i:
        words.append(j)
print(words)
print(label_encoder.inverse_transform(predictions))
