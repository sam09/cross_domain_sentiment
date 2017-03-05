from pre_process import termdocumentmatrix
from keras.models import Sequential
from keras.layers import Dense



def create_model(l):
	model = Sequential()
	model.add(Dense(l, input_dim=l, init='uniform', activation='relu'))
	model.add(Dense(l/2, init='uniform', activation='relu'))
	model.add(Dense(1, init='uniform', activation='tanh'))


	model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
	return model



TRAIN_DIR = "Data/train/"
TEST_DIR = "Data/test/"
UNLABELED_DIR = "Data/"

output = "NN_OUTPUT.txt"
names = ["Book", "Kitchen", "Electronics", "DVD"]

write_str = "Training Domain\tTarget Domain\tAccuracy on Labeled Data\n"
def get_accuracy(pred, testY):

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
		model = create_model(l)

		# train
		model.fit(trainX, trainY, nb_epoch=15, batch_size=100)
		pred = model.predict(testX)
		pred = [round(x) for x in pred]
		
		test_accuracy = get_accuracy(pred, testY)
		print i, j, test_accuracy
		write_str += str(i)+"\t"+str(j)+"\t"+str(test_accuracy) + "\n"

f = open(output, "w")
f.write(write_str)
f.close()