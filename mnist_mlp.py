# organize imports
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.datasets import mnist
from keras.utils import np_utils
import tensorflowjs as tfjs

# fix a random seed for reproducibility
np.random.seed(9)

# user inputs
nb_epoch            = 25
num_classes         = 10
batch_size          = 64
train_size          = 60000
test_size           = 10000
v_length            = 784
model_save_path     = "output"

# split the mnist data into train and test
(trainData, trainLabels), (testData, testLabels) = mnist.load_data()

# reshape and scale the data
trainData   = trainData.reshape(train_size, v_length)
testData    = testData.reshape(test_size, v_length)
trainData   = trainData.astype("float32")
testData    = testData.astype("float32")
trainData  /= 255
testData   /= 255

# convert class vectors to binary class matrices --> one-hot encoding
mTrainLabels  = np_utils.to_categorical(trainLabels, num_classes)
mTestLabels   = np_utils.to_categorical(testLabels, num_classes)

# create the MLP model
model = Sequential()
model.add(Dense(512, input_shape=(v_length,)))
model.add(Activation("relu"))
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation("softmax"))

# compile the model
model.compile(loss="categorical_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

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