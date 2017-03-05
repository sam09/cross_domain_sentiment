from gensim.models import Doc2Vec
from gensim.utils import to_unicode
from collections import namedtuple
import numpy as np
from sklearn.linear_model import LinearRegression
from random import shuffle
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec

train_path = "Data/train/BookTrain.csv"
test_path = "Data/test/DVDTest.csv"

SentimentDocument = namedtuple("SentimentDocument", "words tags sentiment split")

alldocs = []

def sentiment_to_int(i):
	if i == "pos":
		return 1
	return 0


def get_data(file_path, split="train"):
	f = open(file_path)
	f.readline()
	k = 0
	for i in f.readlines():
		s = i.split(",")
		words = s[1]
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
	c = c*1.0
	c/= len(testY)
   	return c

get_data(train_path)
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
models_by_name.append(ConcatenatedDoc2Vec([simple_models[1], simple_models[0]]))


doc_list = alldocs[:]

passes = 20

for epoch in range(passes):
	shuffle(doc_list)
	for i in models_by_name:
		model.train(doc_list)

for model in models_by_name:
	trainX, trainY  = split(train_docs, model)
	testX, testY = split(test_docs, model)

	linear_model = LinearRegression()
	linear_model.fit(trainX, trainY)
		
	pred = linear_model.predict(testX)
	test_accuracy = get_accuracy(pred, testY)

	print "Epoch", model, test_accuracy