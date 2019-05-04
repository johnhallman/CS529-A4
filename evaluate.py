import numpy as np
import pandas as pd
import os
import sys
import pickle
import re
import json
import matplotlib.pyplot as plt

from tensorflow.python.client import device_lib
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Activation, Input, Dropout, \
    Add, add, LSTM, Bidirectional, concatenate
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
	w_to_i = {w:i for i, w in enumerate(vocabulary)} 

	embedding = np.zeros(shape=(len(vocabulary), glove_dimension))
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

def test_preprocess(data, features, glove, vocabulary):
	w_to_i = {w:i for i, w in enumerate(vocabulary)}     
	embedding = np.zeros(shape=(len(vocabulary), glove_dimension))
	for w, i in w_to_i.items(): embedding[i,:] = glove[w]

	x_img, x_word, y = [], [], []
	for k, v in data.items():
		x_img.append(features[v['image_id']])
		x_word.append(np.array([w_to_i[w] for w in v['q']]))
		y.append(v['a'])
	x_word = sequence.pad_sequences(x_word, maxlen=25)
	x = [np.array(x_img), np.array(x_word)]
	return x, y

def model_output(model, x_input, answer_key):
	y_index = model.predict(x_input)
	return [answer_key[np.argmax(pred)] for pred in y_index]

def evaluate(model, x_test, y_test, i_to_a):
    y_pred = model_output(model, x_test, i_to_a)
    score = np.mean([pred == truth for pred, truth in zip(y_pred, y_test)])
    return score

def model_answer_json(model, x_input, q_ids, answer_key):
	y_index = model.predict(x_input)
	preds = [answer_key[np.argmax(pred)] for pred in y_index]
	d = [{'answer':p, 'question_id':i} for p, i in zip(preds, q_ids)]
	with open(file_name, 'w') as f:
		json.dump(d, f)

def ultimate_evaluate(model, x_test, y_test, q_ids, i_to_a):
    y_index = model.predict(x_input)
    y_pred = [answer_key[np.argmax(pred)] for pred in y_index]
    score = np.mean([pred == truth for pred, truth in zip(y_pred, y_test)])
    d = [{'answer':p, 'question_id':i} for p, i in zip(y_pred, q_ids)]
    with open(file_name, 'w') as f:
        json.dump(d, f)
    return score

if __name__=="__main__":

	print("\n---- Starting Training ----\n")

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

	answer_list, _ = top_answers(train_raw, k=2000)
	train_data, test_data, vocabulary = minimize_vocab(train_raw, test_raw, glove)

	x_train, y_train, embedding, a_to_i = preprocessing(train_data, train_features, glove, answer_list, vocabulary)
	i_to_a = {i:a for a,i in a_to_i.items()}

	x_test, y_test = test_preprocess(test_data, val_features, glove, vocabulary)

	print(" - Data preprocessing completed")


	"""

	print("\n - Evaluating models...")

	concat_15 = load_model("models/concat-model-15.hdf5")
	score_15 = evaluate(concat_15, x_test, y_test, i_to_a)

	print("Concat 15 score: {}". format(score_15))

	concat_10 = load_model("models/concat-model-10.hdf5")
	score_10 = evaluate(concat_10, x_test, y_test, i_to_a)

	print("Concat 10 score: {}". format(score_10))

	"""

	model_answer_json(model, x_input, q_ids, answer_key)





