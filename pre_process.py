import textmining
import pandas as pd
import os
import nltk
from nltk.corpus import sentiwordnet as swn


TRAIN_DIR = "./Dataset/train"

def termdocumentmatrix(filename):
    tdm = textmining.TermDocumentMatrix()

    data = open(TRAIN_DIR + "/" + filename)
    data.readline()

    docs = data.readlines()
    for doc in docs:
        doc_str = doc.split(",")[1]
        tdm.add_doc(doc_str)

    tdm_rows = [x for x in tdm.rows()]
    words = tdm_rows[0]
    word_index_dict = {}
    for i in range(0, len(words)):
        word_index_dict[words[i]] = i
    
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
                    tdm_rows[doc][word_index] *= score
                except:
                    continue

    data.close()
    return tdm_rows
    
    # Write out the matrix to a csv file. Note that setting cutoff=1 means
    # that words which appear in 1 or more documents will be included in
    # the output (i.e. every word will appear in the output). The default
    # for cutoff is 2, since we usually aren't interested in words which
    # appear in a single document. For this example we want to see all
    # words however, hence cutoff=1.
    
    #tdm.write_csv('matrix.csv', cutoff=1)
    

def get(i):
    if 'NN' in i:
        return 'n'
    elif 'VB' in i:
        return 'v'
    elif 'JJ' in i:
        return 'a'
    elif 'RB' in i:
        return 'r'

tdm = termdocumentmatrix("BookTrain.csv")
#print scores

