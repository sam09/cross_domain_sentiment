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
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec

train_path = ["Data/train/DVDTrain.csv", "Data/train/KitchenTrain.csv", "Data/ElectronicsUnlabel.csv" "Data/train/DVDTrain.csv"]

test_path = "Data/train/ElectronicsTrain.csv"
print train_path, test_path
SentimentDocument = namedtuple("SentimentDocument", "words tags sentiment split")

alldocs = []

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
	for i in f.readlines():
		s = i.split(",")
		words = ""
		for i in s[1:]:
			words += " " + i.strip().lower()

		words = words.translate(None, string.punctuation)
		tags =[split + "_" + str(k)]
		sentiment = sentiment_to_int(s[0])
		words = to_unicode(words).split()
		alldocs.append(SentimentDocument(words, tags, sentiment, split))
		k+=1

def split(docs, model):
	X = []
	Y = []
	for i in docs:
		Y.append(i.sentiment)
		X.append(model.docvecs[i.tags[0]])
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

get_data(train_path[0])
#get_data(train_path[2])
get_data(train_path[1])

get_data(test_path, split="test")

train_docs = [doc for doc in alldocs if doc.split == 'train']
test_docs = [doc for doc in alldocs if doc.split == 'test']

simple_models = [
    # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
    Doc2Vec(dm=1, dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=2),
    # PV-DBOW 
    Doc2Vec(dm=0, size=100, negative=5, hs=0, min_count=2),
    # PV-DM w/average
    Doc2Vec(dm=1, dm_mean=1, size=100, window=10, negative=5, hs=0, min_count=2)
]

for model in simple_models:
	model.build_vocab(alldocs)

models_by_name = []

models_by_name.append(ConcatenatedDoc2Vec([simple_models[1], simple_models[2]]))


doc_list = alldocs[:]

passes = 10

for epoch in range(passes):
	shuffle(doc_list)
	for i in models_by_name:
		model.train(doc_list)
	print epoch,  " Done" 

#training with SVC

for model in models_by_name:
	trainX, trainY  = split(train_docs, model)
	testX, testY = split(test_docs, model)
	linear_model = SVC(probability=True)
	linear_model.fit(trainX, trainY)
	pred = linear_model.predict_proba(testX)
	c = 0
	tot =0
	for i in range(0, len(pred)):
		if pred[i][0] >= 0.60 or pred[i][0] <=0.40:
			tot += 1
		if pred[i][0] > 0.5 and testY[i] == 0:
			c+=1
		elif testY[i]==1 and pred[i][0] <0.5:
			c+=1

	#test_accuracy = get_accuracy(pred, testY)
	c*=1.0
	tot *= 1.0
	test_accuracy = c/tot
	print test_accuracy

#training with CNN
"""
for model in models_by_name:
	trainX, trainY  = split(train_docs, model)
	testX, testY = split(test_docs, model)
	trainX = np.asarray(trainX)
	testX = np.asarray(testX)
	trainY = np.asarray(trainY)

	trainX = reshapeX(trainX)
	testX = reshapeX(testX)
	trainY = reshapeY(trainY)
	print len(trainX[0])
	cnn = CNN(200).get_model()
	cnn.fit(trainX, trainY,  nb_epoch=200, batch_size=250)
	pred = cnn.predict(testX)
	pred = np_utils.probas_to_classes(pred)
	test_accuracy = get_accuracy(pred, testY)
	print test_accuracy
"""