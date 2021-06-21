# Par Belkacem Mohammed
# Publié sous licence CC0
# Copyright Juillet 2021

import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import np_utils
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.models import model_from_yaml
from pickle import dump
sentences=[]
#Construction du texte global
for ligne in open("corpus-kab.txt",encoding='utf-8'):
        sentence=[]
        line=ligne.strip().split()

        for i in line:
            j=i.split('/')
            couple=(j[0],j[1])
            sentence.append(couple)
        sentences.append(sentence)


tags = set([
    tag for sentence in sentences
    for _, tag in sentence
])

train_test_cutoff = int(.80 * len(sentences))

training_sentences = sentences[:train_test_cutoff]
testing_sentences = sentences[train_test_cutoff:]

train_val_cutoff = int(.25 * len(training_sentences))
validation_sentences = training_sentences[:train_val_cutoff]
training_sentences = training_sentences[train_val_cutoff:]


def add_basic_features(sentence_terms, index):
    """ Compute some very basic word features.
        :param sentence_terms: [w1, w2, ...]
        :type sentence_terms: list
        :param index: the index of the word
        :type index: int
        :return: dict containing features
        :rtype: dict
    """
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
        #'next2_word': '' if index == len(sentence_terms) - 1 else sentence_terms[index + 2],
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

def untag(tagged_sentence):
    """
    Remove the tag for each tagged term.
:param tagged_sentence: a POS tagged sentence
    :type tagged_sentence: list
    :return: a list of tags
    :rtype: list of strings
    """
    return [w for w, _ in tagged_sentence]

def transform_to_dataset(tagged_sentences):
    X, y = [], []

    for pos_tags in tagged_sentences:
        for index, (term, class_) in enumerate(pos_tags):
            # Add basic NLP features for each sentence term
            X.append(add_basic_features(untag(pos_tags), index))
            y.append(class_)
    return X, y
X_train, y_train = transform_to_dataset(training_sentences)
X_test, y_test = transform_to_dataset(testing_sentences)
X_val, y_val = transform_to_dataset(validation_sentences)

dict_vectorizer = DictVectorizer(sparse=False)
dict_vectorizer.fit(X_train + X_test + X_val)


# Convert dict features to vectors
X_train = dict_vectorizer.transform(X_train)

X_test = dict_vectorizer.transform(X_test)
X_val = dict_vectorizer.transform(X_val)

# Fit LabelEncoder with our list of classes
label_encoder = LabelEncoder()
label_encoder.fit(y_train + y_test + y_val)
#Asekles n tecraḍ
dump(label_encoder, open('label_encoder.pkl', 'wb'))


y_train = label_encoder.transform(y_train)

labels=label_encoder.inverse_transform(y_train)
y_test = label_encoder.transform(y_test)
y_val = label_encoder.transform(y_val)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
y_val = np_utils.to_categorical(y_val)


def build_model(input_dim, hidden_neurons, output_dim):
    model = Sequential([

        Dense(hidden_neurons, input_dim=input_dim),
        Activation('relu'),
        Dropout(0.2),
        Dense(hidden_neurons),
        Activation('relu'),
        Dropout(0.2),
        Dense(hidden_neurons),
        Activation('relu'),
        Dropout(0.2),
        Dense(output_dim, activation='softmax')

    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model_params = {
     'build_fn': build_model,
    'input_dim': X_train.shape[1],
    'hidden_neurons': 1024,
    'output_dim': y_train.shape[1],
    'epochs': 8,
    'batch_size': 256,
    'verbose': 1,
    'validation_data': (X_val, y_val),
    'shuffle': True
}


model=build_model(X_train.shape[1],1024, y_train.shape[1])
#asekles n tneɣruft
dump(model, open('model.clf', 'wb'))

clf = KerasClassifier(**model_params)
hist = clf.fit(X_train, y_train)


def plot_model_performance(train_loss, train_acc, train_val_loss, train_val_acc):
    
    blue= '#34495E'
    green = '#2ECC71'
    orange = '#E23B13'

    # plot model loss
    fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 8))
    ax1.plot(range(1, len(train_loss) + 1), train_loss, blue, linewidth=5, label='asekyed')
    ax1.plot(range(1, len(train_val_loss) + 1), train_val_loss, green, linewidth=5, label='asentem')
    ax1.set_xlabel('# epoch')
    ax1.set_ylabel('loss')
    ax1.tick_params('y')
    ax1.legend(loc='upper right', shadow=False)
    ax1.set_title('Taneɣruft n usruḥu simal irennu umḍan n #epochs', color=orange, fontweight='bold')

    # plot model accuracy
    ax2.plot(range(1, len(train_acc) + 1), train_acc, blue, linewidth=5, label='asekyed')
    ax2.plot(range(1, len(train_val_acc) + 1), train_val_acc, green, linewidth=5, label='tiseddi')
    ax2.set_xlabel('# epoch')
    ax2.set_ylabel('accuracy')
    ax2.tick_params('y')
    ax2.legend(loc='lower right', shadow=False)
    ax2.set_title('Taneɣruft n tseddi simal irennu umḍan n #epochs', color=orange, fontweight='bold')
    plt.show()

plot_model_performance(
    train_loss=hist.history.get('loss', []),
    train_acc=hist.history.get('acc', []),
    train_val_loss=hist.history.get('val_loss', []),
    train_val_acc=hist.history.get('val_acc', [])
)

