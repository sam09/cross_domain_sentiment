import string
from collections import namedtuple
import gensim
from nltk.corpus import stopwords

stop = set(stopwords.words("english"))

SentimentDocument = namedtuple('SentimentDocument', 'words tags split sentiment')


def clean_data(train):
	f = open(train)
	f.readline()
	train_text = []
	train_labels = []

	for i in f.readlines():
		s = i.split(",")
		l = s[0]
		j = ",".join(s[1:])
		j = j.strip().lower().translate(string.maketrans("",""), string.punctuation)
		if j != "	":
			train_text.append(j)
			if l == "pos":
				train_labels.append(0)
			else:
				train_labels.append(1)
	return train_text, train_labels

def process_docs(test_text, test_labels):
	
	alldocs = []
	n = len(test_text)
	for i in range(0,n):
	    tokens = gensim.utils.to_unicode(test_text[i]).split()
	    words = [ k for k in tokens if k not in stop]
	    tags = [i]
	    split = "test"
	    sentiment = test_labels[i]
	    alldocs.append(SentimentDocument(words, tags, split, sentiment))

	return	alldocs
