#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np
import pandas

# In[2]:

hasil_prediksi = np.loadtxt('prediksi.txt', dtype=int)
indeks_penyakit = 0
data  = pandas.read_csv('/home/herley/riset/wilmay/INTUITIVE2.csv')
PreProc = data['Hasil_PreProcessing'].values

X = data.Hasil_PreProcessing
Y = data.target

# Dimensi tensor LSTM.
max_words = 500
max_len = 150
embedding_dim = 128

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X)
word_index = tokenizer.word_index
X = tokenizer.texts_to_sequences(X)
X = sequence.pad_sequences(X, maxlen=max_len)
Y = Y.values

dict_list=[]
for x in Y:
	m={}
	colCounter=0
	for k in x:
		s='t'+str(colCounter)
		m[s]=k
		colCounter+=1
	dict_list.append(m)

Y = pandas.DataFrame(dict_list).values
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15,random_state = 42)

X_train = X_train.astype('int32')
Y_train = Y_train.astype('int32')
Y_test = Y_test.astype('int32')
X_test = X_test.astype('int32')

def data_chunks(data):
	hasil = np.array([])
	for x in data:
		datanya = np.split(x,16)[indeks_penyakit]
		if(hasil.size == 0):
			hasil = np.array([datanya])
		else:
			hasil = np.concatenate((hasil,[datanya]))
	return hasil

Y_train = data_chunks(Y_train)
Y_test = data_chunks(Y_test)

benar = 0
for x in range(0, len(Y_test)):
	sebenarnya = Y_test[x]
	prediksi = hasil_prediksi[x]

	if(np.array_equal(sebenarnya,prediksi)):
		benar+=1

print("Akurasi: ", (benar/len(Y_test)))