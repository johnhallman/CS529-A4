import numpy as np
import pandas as pd
import os
import sys
import pickle
import re
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.client import device_lib

from keras import regularizers
from keras.backend import mean
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Activation, Input, Dropout, \
    Add, add, LSTM, Bidirectional, Multiply, concatenate, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

def load_json(json_file):
	json_data = open(json_file)
	data = json.load(json_data)
	json_data.close()
	return data

def load_pickle(pickle_file):
	pickle_data = open(pickle_file, "rb")
	data = pickle.load(pickle_data)
	pickle_data.close()
	return data

def load_glove(directory):
	glove = {}
	with open(directory) as file:
		for line in file:
			words = line.split(' ')
			word, vector = words[0], np.asarray(words[1:], dtype='float32')
			glove[word] = vector
	return glove

def extract_questions(questions, answers):
	data = {}
	for q in questions:
		data[q['question_id']] = {'q': q['question'], 'image_id': q['image_id']}
	for a in answers:
		data[a['question_id']]['a'] = a['multiple_choice_answer']
	return data

def top_answers(data, k=1000):
	answer_list = [v['a'] for k, v in data.items()]
	a_set = set(answer_list)
	w_to_i, i_to_w = {}, {}
	for i, w in enumerate(a_set):
		w_to_i[w] = i
		i_to_w[i] = w
	score = len(a_set) * [0]
	for w in answer_list: score[w_to_i[w]] += 1
	w_sorted = np.flip(np.argsort(score))
	top_a = [i_to_w[i] for i in w_sorted[:k]]
	proportion = np.sum([score[i] for i in w_sorted[:k]]) / len(answer_list)
	return top_a, proportion

def clean_sentence(s, glove):
	s_pre = s.lower().replace("\'s", " ").replace("\'", " ").replace("/", " ").replace("?", " ")
	s_clean = s_pre.replace("-", " ").replace(".", " ").replace(",", " ").replace("\"", " ")
	words = []
	for w in s_clean.split(" "):
		if w in glove: words.append(w)
	if len(words) == 0: return None
	return words

def minimize_vocab(train, test, glove):
	vocabulary = set()
	new_train, new_test = {}, {}
	for k, v in train.items():
		clean = clean_sentence(v['q'], glove)
		if clean == None: continue
		v['q'] = clean
		new_train[k] = v
		vocabulary.update(clean)
	for k, v in test.items():
		clean = clean_sentence(v['q'], glove)
		if clean == None: continue
		v['q'] = clean
		new_test[k] = v
		vocabulary.update(clean)
	return new_train, new_test, vocabulary

def a_to_one_hot(a, a_to_i, answer_length):
	b = np.zeros(answer_length)
	b[a_to_i[a]] = 1
	return b

def preprocessing(data, features, glove, answers, vocabulary):
	answer_length = len(answers)
	a_to_i = {w:i for i, w in enumerate(answers)}
	w_to_i = {w:(i+1) for i, w in enumerate(vocabulary)} 

	embedding = np.zeros(shape=(len(vocabulary) + 1, glove_dimension))
	for w, i in w_to_i.items():
	    embedding[i,:] = glove[w]

	x_img, x_word, y = [], [], []
	for k, v in data.items():
		if v['a'] not in a_to_i: continue
		x_img.append(features[v['image_id']])
		x_word.append(np.array([w_to_i[w] for w in v['q']]))
		y.append(a_to_one_hot(v['a'], a_to_i, answer_length))
	x_word = sequence.pad_sequences(x_word, maxlen=25)
	x = [np.array(x_img), np.array(x_word)] # vstack(np.expand_dims(x_word, axis=0))
	return x, np.array(y), embedding, a_to_i


def model_one(x, y, embedding, filepath, epochs=100, hidden_dim=512):
	(vocab_size, vec_dim) = embedding.shape
	feat_dim, seq_length, output_dim = len(x[0][0]), len(x[1][0]), len(y[0])

	a1 = Input(shape=(feat_dim,))
	b1 = Dense(2 * hidden_dim)(a1)

	a2 = Input(shape=(seq_length,))
	b2 = Embedding(vocab_size, vec_dim, weights=[embedding], input_length=seq_length, trainable=False)(a2)
	#b2 = Bidirectional(LSTM(hidden_dim, return_sequences=True, activation='relu'))(b2)
	b2 = Bidirectional(LSTM(hidden_dim, activation='relu'))(b2)

	b = concatenate([b1, b2])
	c = Dense(output_dim, activation='softmax')(b)

	model = Model(inputs=[a1, a2], outputs=[c])
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	checkpoint = ModelCheckpoint(filepath=filepath+"-{epoch:02d}.hdf5", period=5)
	model.fit(x, y, callbacks=[checkpoint], epochs=epochs, batch_size=64)
	return model


def model_add(x, y, embedding, filepath, epochs=100, hidden_dim=512):
	(vocab_size, vec_dim) = embedding.shape
	feat_dim, seq_length, output_dim = len(x[0][0]), len(x[1][0]), len(y[0])

	a1 = Input(shape=(feat_dim,))
	b1 = Dense(2 * hidden_dim)(a1)

	a2 = Input(shape=(seq_length,))
	b2 = Embedding(vocab_size, vec_dim, weights=[embedding], input_length=seq_length, trainable=False)(a2)
	#b2 = Bidirectional(LSTM(hidden_dim, return_sequences=True, activation='relu'))(b2)
	b2 = Bidirectional(LSTM(hidden_dim, activation='relu'))(b2)

	b = Add()([b1, b2])
	c = Dense(output_dim, activation='softmax')(b)

	model = Model(inputs=[a1, a2], outputs=[c])
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	checkpoint = ModelCheckpoint(filepath=filepath+"-{epoch:02d}.hdf5", period=5)
	model.fit(x, y, callbacks=[checkpoint], epochs=epochs, batch_size=64)
	return model


def model_no_embedding(x, y, embedding, filepath, epochs=100, hidden_dim=512):
	(vocab_size, vec_dim) = embedding.shape
	feat_dim, seq_length, output_dim = len(x[0][0]), len(x[1][0]), len(y[0])
	vec_dim = 50 # train vectors

	a1 = Input(shape=(feat_dim,))
	b1 = Dense(2 * hidden_dim)(a1)

	a2 = Input(shape=(seq_length,))
	b2 = Embedding(vocab_size, vec_dim, weights=[embedding], input_length=seq_length, trainable=False)(a2)
	#b2 = Bidirectional(LSTM(hidden_dim, return_sequences=True, activation='relu'))(b2)
	b2 = LSTM(hidden_dim, activation='relu')(b2)

	b = concatenate([b1, b2])
	c = Dense(output_dim, activation='softmax')(b)

	model = Model(inputs=[a1, a2], outputs=[c])
	opt = Adam(lr=0.0005)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

	checkpoint = ModelCheckpoint(filepath=filepath+"-{epoch:02d}.hdf5", period=1)
	model.fit(x, y, callbacks=[checkpoint], epochs=epochs, batch_size=64)
	return model


def model_final(x, y, embedding, filepath, epochs=100, hidden_dim=512):
	(vocab_size, vec_dim) = embedding.shape
	feat_dim, seq_length, output_dim = len(x[0][0]), len(x[1][0]), len(y[0])

	a1 = Input(shape=(feat_dim,))
	b1 = Dense(2 * hidden_dim, activation='relu')(a1)
	b1 = Dropout(0.2)(b1)
	b1 = Dense(2 * hidden_dim, activation='tanh', activity_regularizer=regularizers.l2(0.01))(b1)

	a2 = Input(shape=(seq_length,))
	b2 = Embedding(vocab_size, vec_dim, weights=[embedding], input_length=seq_length, trainable=False)(a2)
	#b2 = LSTM(hidden_dim, activation='tanh', return_sequences=True)(b2)
	b2 = LSTM(2 * hidden_dim, activation='tanh')(b2)

	b = Multiply()([b1, b2])
	c = Dense(2 * hidden_dim, activation='relu')(b)
	c = Dropout(0.2)(c)
	c = Dense(output_dim, activation='softmax')(b)

	model = Model(inputs=[a1, a2], outputs=[c])
	opt = Adam(lr=0.0005)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

	checkpoint = ModelCheckpoint(filepath=filepath+"-{epoch:02d}.hdf5", period=1)
	model.fit(x, y, callbacks=[checkpoint], epochs=epochs, batch_size=64)
	return model


def model_new_final(x, y, embedding, filepath, epochs=100, hidden_dim=512):
	(vocab_size, vec_dim) = embedding.shape
	feat_dim, seq_length, output_dim = len(x[0][0]), len(x[1][0]), len(y[0])

	a1 = Input(shape=(feat_dim,))
	b1 = Dense(hidden_dim, activation='tanh')(a1)

	a2 = Input(shape=(seq_length,))
	b2 = Embedding(vocab_size, vec_dim, weights=[embedding], input_length=seq_length, trainable=False)(a2)
	#b2 = LSTM(hidden_dim, activation='tanh', return_sequences=True)(b2)
	b2 = LSTM(hidden_dim, activation='tanh')(b2)

	b = Multiply()([b1, b2])
	c = Dense(2 * hidden_dim, activation='relu')(b)
	c = Dropout(0.2)(c)
	c = Dense(output_dim, activation='softmax')(b)

	model = Model(inputs=[a1, a2], outputs=c)
	opt = Adam()
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

	checkpoint = ModelCheckpoint(filepath=filepath+"-{epoch:02d}.hdf5", period=1)
	model.fit(x, y, callbacks=[checkpoint], epochs=epochs, batch_size=64, verbose=2)
	return model


def test_preprocess(data, features, glove, vocabulary):
	w_to_i = {w:i for i, w in enumerate(vocabulary)}     
	embedding = np.zeros(shape=(len(vocabulary), glove_dimension))
	for w, i in w_to_i.items(): embedding[i,:] = glove[w]
	    
	x_img, x_word, y, q_id = [], [], [], []
	for k, v in data.items():
		x_img.append(features[v['image_id']])
		x_word.append(np.array([w_to_i[w] for w in v['q']]))
		y.append(v['a'])
		q_id.append(k) # k is question id!
	x_word = sequence.pad_sequences(x_word, maxlen=25)
	x = [np.array(x_img), np.array(x_word)]
	return x, y, q_id


def model_output(model, x_input, answer_key):
	y_index = model.predict(x_input)
	return [answer_key[np.argmax(pred)] for pred in y_index]

def evaluate(y_pred, y_test):
	return np.mean([pred == truth for pred, truth in zip(y_pred, y_test)])

def write_results(results, file_name):
	with open(file_name, 'w') as f:
		for item in results:
			f.write("%s\n" % item)

def ultimate_evaluate(file_name, model, x_test, y_test, q_ids, answer_key):
	y_index = model.predict(x_test)
	y_pred = [answer_key[np.argmax(pred)] for pred in y_index]
	score = np.mean([pred == truth for pred, truth in zip(y_pred, y_test)])
	d = [{'answer':p, 'question_id':i} for p, i in zip(y_pred, q_ids)]
	with open(file_name, 'w') as f:
		json.dump(d, f)
	return score

if __name__ == "__main__":


	print("\n---- Test ----\n")
	train_features = load_pickle("data/train_features.pickle")
	val_features = load_pickle("data/val_features.pickle")
	train_questions = load_json("data/train_questions.json")['questions']
	val_questions = load_json("data/val_questions.json")['questions']
	train_answers = load_json("data/train_annotations.json")['annotations']
	val_answers = load_json("data/val_annotations.json")['annotations']

	print(" - Raw data finished loading")

	glove_dimension = 100
	glove_directory = "glove.6B/glove.6B." + str(glove_dimension) + "d.txt"
	glove = load_glove(glove_directory)

	print(" - Glove finished loading")

	train_raw = extract_questions(train_questions, train_answers)
	test_raw = extract_questions(val_questions, val_answers)

	answer_list, _ = top_answers(train_raw, k=500)
	train_data, test_data, vocabulary = minimize_vocab(train_raw, test_raw, glove)

	x_train, y_train, embedding, a_to_i = preprocessing(train_data, train_features, glove, answer_list, vocabulary)
	i_to_a = {i:a for a,i in a_to_i.items()}

	x_test, y_test, q_test = test_preprocess(test_data, val_features, glove, vocabulary)

	print(" - Data preprocessing completed")

	x_empty_test = [np.zeros(shape=x_test[0].shape), x_test[1]]

	model_last = load_model("finalfinal-model-06.hdf5")
	#model_last = model_new_final(x_train, y_train, embedding, "final-model", epochs=4, hidden_dim=512)
	#y_pred_concat = model_output(model_no_embed, x_test, i_to_a)
	score_concat = ultimate_evaluate("empty_preds.json", model_last, x_empty_test, y_test, q_test, i_to_a)
	print("No image score: {}".format(score_concat))

	print(" - Concat model finished training and evaluation")
	


	print("\n---- Program Complete ----\n")









