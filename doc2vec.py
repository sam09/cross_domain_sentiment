from gensim.models import Doc2Vec
from gensim.utils import to_unicode
from collections import namedtuple
from cnn import CNN
import string
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from keras.utils import np_utils
from random import shuffle
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
SentimentDocument = namedtuple("SentimentDocument", "words tags sentiment split")

train = [
		 "./Data/train/BookTrain.csv",
	     "./Data/train/DVDTrain.csv", 
		 "./Data/train/ElectronicsTrain.csv",
		 "./Data/train/KitchenTrain.csv"
]
test = ["./Data/test/BookTest.csv","./Data/test/DVDTest.csv","./Data/test/ElectronicsTest.csv","./Data/test/KitchenTest.csv"]
unlabel = ["./Data/BookUnlabel.csv", "./Data/DVDUnlabel.csv", "./Data/ElectronicsUnlabel.csv", "./Data/KitchenUnlabel.csv"]

test_path = unlabel[1]

def sentiment_to_int(i):
	if i == "pos":
		return 1
	return 0

def reshapeX(X):
	n = len(X)
	t = len(X[0])
	return X.reshape(n, 1, t, 1)

def reshapeY(Y):
	return np_utils.to_categorical(Y)


def get_data(file_path, split="train"):
	f = open(file_path)
	f.readline()
	k = 0
	docs = []
	for i in f.readlines():
		s = i.split(",")
		words = ""
		for i in s[1:]:
			words += " " + i.strip().lower()

		words = words.translate(None, string.punctuation)
		tags =[split + "_" + str(k)]
		sentiment = sentiment_to_int(s[0])
		words = to_unicode(words).split()
		docs.append(SentimentDocument(words, tags, sentiment, split))
		k+=1
	f.close()
	return docs

def split(docs, model):
	X = []
	Y = []
	for i in docs:
		Y.append(i.sentiment)
		X.append(model.docvecs[i.tags[0]])
	return X, Y

def split_test(docs, model):
	X = []
	Y = []
	for i in docs:
		Y.append(i.sentiment)
		X.append(model.infer_vector(i.words))
	return X, Y

def get_accuracy(pred, testY):
	c = 0
	pred = [round(x) for x in  pred]
	for i in range(0, len(pred)):
		if pred[i] == testY[i]:
			c+=1
	print c
	c = c*1.0
	c/= len(testY)
	print len(testY)
   	return c


simple_models = [
    # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
    Doc2Vec(dm=1, dm_concat=1, vector_size=100, window=5, negative=5, hs=0, min_count=2, epochs=55),
		    # PV-DBOW 
	Doc2Vec(dm=0, vector_size=100, negative=5, hs=0, min_count=2, epochs=100),
		    # PV-DM w/average
	Doc2Vec(dm=1, dm_mean=1, vector_size=100, window=10, negative=10, hs=0, min_count=2, epochs=55)
]


train_docs = []
for i in range(len(train)):
	train_docs.extend(get_data(train[i]))

test_docs = get_data(test_path, split="test")

simple_models[1].build_vocab(train_docs)
models_by_name =[]

#models_by_name.append(simple_models[0])
models_by_name.append(simple_models[1])
#models_by_name.append(simple_models[2])
#models_by_name.append(ConcatenatedDoc2Vec([simple_models[1], simple_models[2]]))

doc_list = train_docs
shuffle(doc_list)
for i in range(len(models_by_name)):
	if i<3:
		models_by_name[i].train(doc_list, epochs=models_by_name[i].epochs, total_examples=models_by_name[i].corpus_count)
	else:
		models_by_name[i].train(doc_list)


for train_i in range(len(train)):
	for test_i in range(4):
		train_path = train[train_i]
		train_docs = get_data(train_path)

		simple_models = [
		    # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
		    Doc2Vec(dm=1, dm_concat=1, vector_size=100, window=5, negative=5, hs=0, min_count=2, epochs=55),
		    # PV-DBOW 
		    Doc2Vec(dm=0, vector_size=100, negative=5, hs=0, min_count=2, epochs=100),
		    # PV-DM w/average
		    Doc2Vec(dm=1, dm_mean=1, vector_size=100, window=10, negative=10, hs=0, min_count=2, epochs=55)
		]

		simple_models[1].build_vocab(train_docs)
		models_by_name =[]

		#models_by_name.append(simple_models[0])
		models_by_name.append(simple_models[1])
		#models_by_name.append(simple_models[2])
		#models_by_name.append(ConcatenatedDoc2Vec([simple_models[1], simple_models[2]]))


		doc_list = train_docs
		shuffle(doc_list)
		for i in range(len(models_by_name)):
			if i<3:
				models_by_name[i].train(doc_list, epochs=models_by_name[i].epochs, total_examples=models_by_name[i].corpus_count)
			else:
				models_by_name[i].train(doc_list)

		for test_j in range(2):
			test_path = test[test_i]
			title = "Test Data"
			if test_j==1:
				test_path = unlabel[test_i]
				title = "Cross  Validation Set(Larger Data Set)"

			print title
			test_docs = get_data(test_path, split="test")
			for model in models_by_name:
				
				testX, testY = split_test(test_docs, model)
				trainX, trainY  = split(train_docs, model)
				
				trainX = np.asarray(trainX)
				testX = np.asarray(testX)
				trainY = np.asarray(trainY)
				testY = np.asarray(testY)
				trainX = reshapeX(trainX)
				testX = reshapeX(testX)
				trainY = reshapeY(trainY)
				cnn = CNN(100).get_model()
				cnn.fit(trainX, trainY,  epochs=20, batch_size=250, verbose=0)
				pred = cnn.predict(testX)
				val = []
				for i in pred:
					if i[0] > 0.50:
						val.append(0)
					else:
						val.append(1)
				
				linear_model = SVC()
				linear_model.fit(trainX, trainY)
				val = linear_model.predict(testX)
				test_accuracy = precision_recall_fscore_support(testY, val, average="weighted")

				print "Accuracy = {0:.4f}".format( accuracy_score(testY, val))
				print "Precision = {0:.4f}".format(test_accuracy[0])
				print "Recall = {0:.4f}".format(test_accuracy[1])
				print "F1-Score = {0:.4f}".format(test_accuracy[2])

"""
#training with SVC
for model in models_by_name:
	trainX, trainY  = split(train_docs, model)
	testX, testY = split_test(test_docs, model)
	linear_model = SVC()
	linear_model.fit(trainX, trainY)
	pred = linear_model.predict(testX)
	print accuracy_score(testY, pred)
	print precision_recall_fscore_support(testY, pred, average="weighted")

for model in models_by_name:
	trainX, trainY  = split(train_docs, model)
	testX, testY = split_test(test_docs, model)
	trainX = np.asarray(trainX)
	testX = np.asarray(testX)
	trainY = np.asarray(trainY)
	testY = np.asarray(testY)

	trainX = reshapeX(trainX)
	testX = reshapeX(testX)
	trainY = reshapeY(trainY)
	cnn = CNN(100).get_model()
	cnn.fit(trainX, trainY,  epochs=20, batch_size=250, verbose=0)
	pred = cnn.predict(testX)
	val = []
	for i in pred:
		if i[0] > 0.50:
			val.append(0)
		else:
			val.append(1)
	test_accuracy = precision_recall_fscore_support(testY, val, average="weighted")
	print train, test_path
	print "Accuracy = {0:.4f}".format( accuracy_score(testY, val))
	print "Precision = {0:.4f}".format(test_accuracy[0])
	print "Recall = {0:.4f}".format(test_accuracy[1])
	print "F1-Score = {0:.4f}".format(test_accuracy[2])
"""
