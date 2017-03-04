from pre_process import termdocumentmatrix
from cnn import CNN
from keras.utils import np_utils


TRAIN_DIR = "Data/train/"
TEST_DIR = "Data/test/"
UNLABELED_DIR = "Data/"

output = "TDM_OUTPUT.txt"
names = ["Book"]#, "Kitchen", "Electronics", "DVD"]

f = open(output, "w")
f.write("Training Domain\tTarget Domain\tAccuracy on Labeled Data\tAccuracy on Unlabeled Data\n")
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

for i in names:
	for j in names:
		train_path = TRAIN_DIR + i + "Train.csv"
		test_path = TEST_DIR + j + "Test.csv"

		trainY, trainX, testY, testX, words  = termdocumentmatrix(train_path, test_path)
		n = len(trainX[0][0])

		model = CNN(n).get_model()

		# train
		model.fit(trainX, trainY, nb_epoch=20, batch_size=20)
		
		test_accuracy = get_accuracy(model, testX, testY)
		write_str = str(i)+"\t"+str(j)+"\t"+str(test_accuracy)  + "\n"
		f.write(write_str)

f.close()