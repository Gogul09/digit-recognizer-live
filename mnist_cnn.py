# organize imports
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.datasets import mnist
from keras.utils import np_utils
import tensorflowjs as tfjs

# fix a random seed for reproducibility
np.random.seed(9)

# user inputs
nb_epoch            = 10
num_classes         = 10
batch_size          = 200
train_size          = 60000
test_size           = 10000
model_save_path     = "output/cnn"

# split the mnist data into train and test
(trainData, trainLabels), (testData, testLabels) = mnist.load_data()

# reshape and scale the data
trainData	= trainData.reshape(trainData.shape[0], 28, 28, 1)
testData	= testData.reshape(testData.shape[0], 28, 28, 1)
trainData   = trainData.astype("float32")
testData    = testData.astype("float32")
trainData  /= 255
testData   /= 255

# convert class vectors to binary class matrices --> one-hot encoding
mTrainLabels  = np_utils.to_categorical(trainLabels, num_classes)
mTestLabels   = np_utils.to_categorical(testLabels, num_classes)

# create the MLP model
model = Sequential()
model.add(Convolution2D(32, (5, 5), border_mode='valid', input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# compile model
model.compile(loss='categorical_crossentropy', 
			  optimizer='adam', 
			  metrics=['accuracy'])

# fit the model
history = model.fit(trainData, 
                    mTrainLabels,
                    validation_data=(testData, mTestLabels),
                    batch_size=batch_size,
                    nb_epoch=nb_epoch,
                    verbose=2)

# evaluate the model
scores = model.evaluate(testData, mTestLabels, verbose=0)

# print the results
print ("[INFO] test score - {}".format(scores[0]))
print ("[INFO] test accuracy - {}".format(scores[1]))

# save tf.js specific files in model_save_path
tfjs.converters.save_keras_model(model, model_save_path)