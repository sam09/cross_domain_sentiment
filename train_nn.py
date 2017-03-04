from pre_process import termdocumentmatrix
from keras.models import Sequential
from keras.layers import Dense



def create_model(n,l):
	model = Sequential()
	model.add(Dense(n, input_dim=l, init='uniform', activation='relu'))
	model.add(Dense(n/2, init='uniform', activation='relu'))
	model.add(Dense(1, init='uniform', activation='tanh'))


	model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
	return model



TRAIN_DIR = "Data/train/"
TEST_DIR = "Data/test/"
UNLABELED_DIR = "Data/"

output = "NN_OUTPUT.txt"
names = ["Book"]#, "Kitchen", "Electronics", "DVD"]

write_str = "Training Domain\tTarget Domain\tAccuracy on Labeled Data\tAccuracy on Unlabeled Data\n"
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

		trainY, trainX, testY, testX, words = termdocumentmatrix(train_path=train_path, test_path=test_path, cnn=False)
		l = len(trainX[0])
		n = len(trainX)
		print n, l
		model = create_model(n,l)

		# train
		model.fit(trainX, trainY, nb_epoch=20, batch_size=500)
		
		test_accuracy = get_accuracy(model, testX, testY)	
		write_str += str(i)+"\t"+str(j)+"\t"+str(test_accuracy) + "\n"

f = open(output, "w")
f.write(write_str)
f.close()