from pre_process import termdocumentmatrix
from keras.models import Sequential
from keras.layers import Dense



def create_model(n):
	model = Sequential()
	model.add(Dense(n, input_dim=n, init='uniform', activation='relu'))
	model.add(Dense(n/2, init='uniform', activation='relu'))
	model.add(Dense(1, init='uniform', activation='tanh'))


	model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
	return model

"""
Create a CNN
Feed Training Data
Predict Test Data
End Result Done
"""

TRAIN_DIR = "Data/train/"
TEST_DIR = "Data/test/"
UNLABELED_DIR = "Data/"

output = "NN_OUTPUT.txt"
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
		unlabeled_path = UNLABELED_DIR + j + "Unlabel.csv"

		trainY, trainX, testY, testX, UnY, UnX  = termdocumentmatrix(train_path, test_path, unlabeled_path)
		n = len(trainX[0][0])

		model = create_model(n)

		# train
		model.fit(trainX, trainY, nb_epoch=20, batch_size=20)
		
		test_accuracy = get_accuracy(model, testX, testY)	
		unlabeled_accuracy = get_accuracy(model, UnX, UnY)
		write_str = str(i)+"\t"+str(j)+"\t"+str(test_accuracy) + "\t" + str(unlabeled_accuracy) + "\n"
		f.write(write_str)

f.close()