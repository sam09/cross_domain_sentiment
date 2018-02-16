from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, AveragePooling2D
from keras.utils import np_utils
from keras.models import Sequential


class CNN:

	def __init__(self, size):
		self.size = size

	def get_model(self):
		cnn = Sequential()
		cnn.add(Convolution2D(50, kernel_size=(3,1), strides=(1,1) , activation="relu", padding="same", input_shape = (1,self.size,1) ))
		cnn.add(Convolution2D(50, kernel_size=(4,1), strides=(1,1) , activation="relu", padding="same", input_shape = (1,self.size,1) ))
		cnn.add(Convolution2D(50, kernel_size=(5,1), strides=(1,1) , activation="relu", padding="same", input_shape = (1,self.size,1) ))
		cnn.add(MaxPooling2D(pool_size=(self.size - 2, 1), strides=(1,1), padding="same" ))
		cnn.add(MaxPooling2D(pool_size=(self.size - 3, 1), strides=(1,1), padding="same" ))
		cnn.add(MaxPooling2D(pool_size=(self.size - 4, 1), strides=(1,1), padding="same" ))
		cnn.add(Flatten())
		cnn.add(Dense(128, activation="softmax"))
		cnn.add(Dropout(0.5))
		cnn.add(Dense(2, activation="softmax"))
		cnn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
		return cnn