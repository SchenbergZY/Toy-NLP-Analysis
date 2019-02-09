from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.models import Model
from keras.layers import Dense, Embedding, Input,GlobalAveragePooling1D,GlobalMaxPool1D,GlobalMaxPool1D,concatenate
from keras.layers import LSTM, Bidirectional, Dropout,GRU
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.preprocessing import text, sequence
import numpy as np
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from itertools import chain
#nltk.download('punkt') #first run with this then not needed anymore
#nltk.download('wordnet')
#nltk.download('stopwords')
#max sentence length 533
#max originwords length 55
#parameter
maxlen = 55
max_features = 8300#8299
embed_size = 300
batch_size = 100
nb_epoch = 5
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-5)

#build model
def make_df(max_features, maxlen):
	data = np.load('1_data.npy')
	#print (data.shape)
	X = data[:,0]
	y = data[:,1]
	#print(np.unique(y))
	i = 0
	j =100
	length = 0
	lemmatizer = WordNetLemmatizer()
	stop_words = set(stopwords.words("english"))
	tokenizer = Tokenizer(num_words=20000)
	X1 = X[y!='DRI_Unspecified']
	y = y[y!='DRI_Unspecified']
	X2 = X1[y!='Sentence']
	y = y[y!='Sentence']
	tokenizer.fit_on_texts(X2)

	vocab_size = len(tokenizer.word_index) + 1
	print('vocabsize',vocab_size)#8299
	#print(tokenizer.word_index)
	print(len(tokenizer.word_index))
	argmaxx = 0
	np.save('word_index.npy', tokenizer.word_index)

	X3 = []
	for x in X2:
	    if (y[i] != 'Sentence') and (y[i] != 'DRI_Unspecified'):
	        print(i)
	        print(x,y[i])
	        words = nltk.word_tokenize(x)
	        no_stp_words = [word for word in words if not word in stop_words]
	        origin_words = []
	        for xx in no_stp_words:
	            if len(xx)<2 and xx != 'R':
	                continue
	            xx = lemmatizer.lemmatize(lemmatizer.lemmatize(xx, wordnet.VERB), wordnet.NOUN)
	            if xx == 'discus':  #fix discuss bug
	                xx = 'discuss'
	            origin_words.append(xx)
	        sequences = tokenizer.texts_to_sequences(origin_words)
	        sequences = list(chain.from_iterable(sequences))
	        if len(sequences)>length:
	            argmaxx = i
	            length = len(sequences)
	        X3.append(sequences)
	    i+=1
	X3 = np.array(X3)
	train_data = pad_sequences(X3,padding='post',maxlen=length)
	word_index = tokenizer.word_index
	y[y=='DRI_Approach'] = 0
	y[y=='DRI_Background'] = 1
	y[y=='DRI_Challenge'] = 2
	y[y=='DRI_Challenge_Hypothesis'] = 2
	y[y=='DRI_Challenge_Goal'] = 2
	y[y=='DRI_FutureWork'] = 3
	y[y=='DRI_Outcome'] = 4
	y[y=='DRI_Outcome_Contribution'] = 4
	y = np_utils.to_categorical(y, num_classes=5)
	print('maxlen',length)
	print('data_loading_finish')
	return train_data, y, word_index

def make_glovevec(glovepath, max_features, embed_size, word_index, veclen=300):
    embeddings_index = {}
    f = open(glovepath)
    for line in f:
        values = line.split()
        word = ' '.join(values[:-300])
        coefs = np.asarray(values[-300:], dtype='float32')
        embeddings_index[word] = coefs.reshape(-1)
    f.close()

    nb_words = max(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim



def BidLstm(maxlen, max_features, embed_size,embedding_matrix):#, embedding_matrix):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size,input_length = maxlen, weights=[embedding_matrix],
                  trainable=False)(inp)
    x = Bidirectional(LSTM(256, return_sequences=True, dropout=0.1,
                           recurrent_dropout=0.1))(x)
    x1_att = Attention(maxlen)(x)
    x = Bidirectional(GRU(256, return_sequences=True, dropout=0.1,
                           recurrent_dropout=0.1))(x)
    x2 = Attention(maxlen)(x)
    x2_att = Attention(maxlen)(x)
    x2_avg = GlobalAveragePooling1D()(x)
    x2_max = GlobalMaxPool1D()(x)
    x = concatenate([x1_att,x2_att,x2_avg,x2_max])
    x = Dense(16, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(5, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)

    return model

#main
X,y,word_index = make_df(max_features, maxlen)
embedding_vector = make_glovevec("glove.840B.300d.txt",max_features, embed_size, word_index)
print('start_modeling')
model = BidLstm(maxlen, max_features, embed_size, embedding_vector)
model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
print('finish_modeling')
model.summary()


print('training')
choose_index= np.array([839,316 ,4045, 3113, 4167, 3036, 3385, 7329, 3696, 1518, 4131, 4651, 5935, 6183, 7156 ,7620,  812, 6863, 1221, 7075, 4136, 7198, 8095, 1097, 1268, 8149, 2839, 2667, 5534, 3843, 6782, 6322, 1416, 2621, 3298, 6243, 7607, 1776, 5789, 4993, 8510,  303, 5961, 3575, 7045, 1656, 5121,  733, 5291, 1260,  606, 6645, 1208, 2251, 4757,  363,7473, 3535, 2841, 3431,  903, 4187, 8366, 1723, 8250, 4869, 1887, 1614, 3413, 5489,7794, 6649, 7529, 2732, 4536, 7345, 1784, 1736,  505, 3813, 6121, 4572, 5772, 1290,2281, 8481, 5110, 7789, 6511, 3135,  367, 6637, 6988, 3057, 6980,  837, 2887, 4137,5659, 1105])
total_index = np.arange(y.shape[0])
rest_index = np.array([x for x in total_index if x not in choose_index])
X_valid = X[choose_index]
y_valid = y[choose_index]
X_train = X[rest_index]
y_train = y[rest_index]
'''
def generator(features, labels, batch_size):
    # Create empty arrays to contain batch of features and labels#
    batch_features = np.zeros((batch_size, features.shape[1]))
    batch_labels = np.zeros((batch_size,5))
    while True:
        for i in range(batch_size):
            # choose random index in features
            index= np.random.choice(len(features),1)
            batch_features[i] = features[index]
            batch_labels[i] = labels[index]
        yield batch_features, batch_labels
model.fit_generator(generator(X_train, y_train, batch_size),
                    steps_per_epoch = X_train.shape[0] / batch_size,
                    validation_data= generator(X_valid, y_valid,batch_size),
                    validation_steps = y_valid.shape[0] / batch_size,
                    epochs=nb_epoch, verbose=1, max_q_size=100,
                    callbacks=[lr_reducer, csv_logger,checkpoint])
'''
model.fit(X_train,y_train,batch_size=batch_size,epochs=nb_epoch,verbose=1)
model.save('good_1')
y_pred = model.predict(X_valid)
y_pred = y_pred.argmax(axis = 1)
y_pred = np_utils.to_categorical(y_pred, num_classes=5)
from sklearn.metrics import f1_score
for i in range(5):
    print(f1_score(y_valid[:,i], y_pred[:,i]))
print('acc',np.sum(y_pred.argmax(axis=1)==y_valid.argmax(axis=1))/100.)
