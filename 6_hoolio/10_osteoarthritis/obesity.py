#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np
import pandas

# In[2]:
print('>>>Osteoarthritis<<<')
indeks_penyakit = 9
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
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.4,random_state = 42, shuffle=True)

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

# Import necessary Keras library.
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop, SGD, Adam
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SpatialDropout1D
from keras.layers.embeddings import Embedding
import keras


# In[9]:
model = Sequential()
model.add(Embedding(max_words,embedding_dim, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
# Modifikasi Layer LSTM dan Hyperparameter LSTM.
# Tambahkan parameter "return sequences" jika ingin menambah layer LSTM. 
model.add(LSTM(500, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
model.add(LSTM(500, return_sequences=True, dropout=0.5))
model.add(LSTM(500, return_sequences=True, dropout=0.5))
model.add(LSTM(500, return_sequences=True, dropout=0.5))
model.add(LSTM(500, dropout=0.2, recurrent_dropout=0.2))

# Gunakan softmax dan categorical_crossentropy untuk klasifikasi multi kelas.
# Tapi karena kita memaksa luaran berupa angka 0 dan 1 maka digunakan step function sebagai fungsi aktivasi. 
# Atau boleh menggunakan softmax, sigmoid, dll, tapi pada tahap akhir harus ada diskritisasi ke arah 0 atau 1.
# Pada contoh ini digunakan softmax dan hard_sigmoid namun tidak keduanya secara bersama-sama.
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Visualisasikan model ini menggunakan graphviz.
print(model.summary())

# Modifikasi juga jumlah epoch dan batch_size. Nilai epoch dan batch_size ini sengaja diset kecil karena eksperimen ini dicoba di laptop. Silahkan gunakan YUKI, HIKARI atau SHIRO (Linux) untuk training sebenarnya. Jika belum ada akun pada pada komputer, saya buatkan.
epochs = 50
batch_size = 1024

# save model.
# Model ini akan otomatis disimpan jika ada improvisasi (validation_loss dalam kasus ini) dibandingkan iterasi sebelumnya.
model_checkpoint=keras.callbacks.ModelCheckpoint('XLSTM{epoch:02d}.h5',save_weights_only=False, save_best_only=True)

# Hilangkan tanda komentar pada history untuk memulai training.
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,callbacks=[EarlyStopping(monitor='acc', patience=15, min_delta=0.0001), model_checkpoint], verbose=2)

# Konversi nilai probabilitas yang paling besar menjadi 1, lainnya 0.
def replace_response(data):
	hasil = None
	for x in data:
		indeks_maksimum = np.argmax(x)
		a = [0,0,0,0]
		a[indeks_maksimum] = 1
		if(hasil is None):
			hasil = np.array([a])
		else:
			hasil = np.concatenate((hasil, np.array([a])), axis=0)
	return hasil

yans = model.predict(X_test)
jawab = replace_response(yans)
np.savetxt('prediksi.txt', jawab, fmt='%d')