import textmining
import numpy as np
import pandas as pd
import os
import nltk
from nltk.corpus import sentiwordnet as swn
from keras.utils import np_utils

TRAIN_DIR = "./Dataset/train"
TEST_DIR =  "./Dataset/test"

def get(i):
    if 'NN' in i:
        return 'n'
    elif 'VB' in i:
        return 'v'
    elif 'JJ' in i:
        return 'a'
    elif 'RB' in i:
        return 'r'

def norm(doc_str):
	if doc_str == "pos":
		return 1
	return 0
def write_csv(tdm, filename):
    f = open(filename, "w")

    for row in tdm:
        for val in range(0, len(row)-1):
        	f.write(str(val) + ",")
        	f.write(str(row[len(row) - 1]) + "\n")

    f.close()

def add_doc(tdm, file_path):
    data = open(file_path)
    data.readline()
    docs = data.readlines()
    train_labels =[]
    for doc in docs:
        doc_str = doc.split(",")[1]
        if doc_str != "":
            tdm.add_doc(doc_str)
            train_labels.append(norm(doc.split(",")[0]))
    data.close()
    return tdm, train_labels

def add_senti_score( file_path, tdm_rows, word_index_dict):

    data = open(file_path)
    data.readline()
    docs = data.readlines()

    for doc in range(0 , len(docs)):
        doc_str = docs[doc].split(",")[1]
        sentences = doc_str.split(".")
        
        for sentence in sentences:
            token = nltk.word_tokenize(sentence)
            tagged = nltk.pos_tag(token)
            for i in range(0, len(tagged)):
                synset = swn.senti_synsets(tagged[i][0], get(tagged[i][1]))
                score = sum([i.pos_score() - i.neg_score() for i in synset])
                try:
                    word_index = word_index_dict[tagged[i][0].lower()]
                    tdm_rows[doc][word_index] = score * tdm_rows[doc][word_index]
                except:
                    continue
    return len(docs), tdm_rows

def termdocumentmatrix(train_path, test_path):
    
    tdm = textmining.TermDocumentMatrix()

    tdm, train_labels = add_doc(tdm, train_path)
    tdm, test_labels = add_doc(tdm, test_path)

    tdm_rows = [x for x in tdm.rows()]
    
    words = tdm_rows[0]
    word_index_dict = {}

    for i in range(0, len(words)):
        word_index_dict[words[i]] = i
    
    train_len, tdm_rows = add_senti_score(train_path, tdm_rows, word_index_dict)
    test_len, tdm_rows = add_senti_score(test_path, tdm_rows, word_index_dict)

    train_tdm = reshapeX(np.asarray(tdm_rows[1:train_len+1]))
    test_tdm = reshapeX(np.asarray(tdm_rows[train_len+1:train_len+test_len+1]))
    
    train_labels = reshapeY(np.asarray(train_labels))
    test_labels = np.asarray(test_labels)

    return train_labels, train_tdm, test_labels, test_tdm, words
    
def reshapeX(X):
	n = len(X)
	t = len(X[0])
	return X.reshape(n, 1, t, 1)

def reshapeY(Y):
	return np_utils.to_categorical(Y)


#tdm = termdocumentmatrix("./Data/train/BookTrain.csv", "./Data/test/DVDTest.csv")
#write_csv(tdm, "matrix.csv")
#print scores