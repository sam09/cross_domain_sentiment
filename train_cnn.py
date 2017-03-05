from pre_process import termdocumentmatrix
from cnn import CNN
from keras.utils import np_utils
import sys

TRAIN_DIR = "Data/train/"
TEST_DIR = "Data/test/"
UNLABELED_DIR = "Data/"

output = "CNN_OUTPUT.txt"


def get_accuracy(model, testX, testY):

	pred = model.predict(testX)
	pred = np_utils.probas_to_classes(pred)

	correct = 0
	for i in range(0, len(pred)):
		if pred[i] == testY[i]:
			correct+=1

	errors = len(testY)-correct
	error_rate = errors/(len(testY)*1.0)
	return 100 * (1.0 - error_rate)




def train(i, j):

	train_path = TRAIN_DIR + i + "Train.csv"
	test_path = TEST_DIR + j + "Test.csv"

	trainY, trainX, testY, testX, words  = termdocumentmatrix(train_path, test_path)
	n = len(trainX[0][0])

	model = CNN(n).get_model()

		# train
	model.fit(trainX, trainY, nb_epoch=15 batch_size=100)
		
	test_accuracy = get_accuracy(model, testX, testY)
	print i, j, test_accuracy
	write_str = str(i)+"\t"+str(j)+"\t"+str(test_accuracy)  + "\n"
		
	f = open(output, "a")
	f.write(write_str)
	f.close()


train_file = sys.argv[1]
test_file = sys.argv[2]

train(train_file, test_file)