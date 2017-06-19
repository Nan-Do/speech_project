import scipy.io.wavfile as wav
import numpy as np
import speechpy
import random

from itertools import chain

# Keras library
from keras.models import Sequential  
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.optimizers import SGD
from keras.utils import np_utils

random.seed(15)

LABELS = ['Apple', 'Kiwi']
DATA_LENGTH = 48
CEPSTRAL = 13

# This function is used to extract the mfcc features from
# a wav file. 
# file_name must be the name of the file
def get_mfcc_from_file(file_name):
    fs, signal = wav.read(file_name)

    # Append the MFCC of the signal
    # This will create frames of 13 elements each 20 ms
    # Add 0's at the end to the to match the same length for all the different 
    # files
    s = speechpy.mfcc(signal, fs, num_cepstral = CEPSTRAL, frame_stride=0.01)
    for _ in xrange(DATA_LENGTH - len(s)):
        s = np.append(s, np.zeros(CEPSTRAL))
    s.shape = (len(s) / CEPSTRAL, CEPSTRAL) 
    return s


# Read the apple and kiwi sound files
# and process them extracting the mfcc features
apple_data = []
kiwi_data = []
for x in xrange(1, 15):
    for file_name in ('apple', 'kiwi'):
        data = kiwi_data
        if file_name == 'apple':
            data = apple_data

        file_name = 'audio/' + file_name
        if x < 10:
            file_name += '0' + str(x) + '.wav'
        else:
            file_name += str(x) + '.wav'

        data.append(get_mfcc_from_file(file_name))

# Split the data for Training and Test
data = np.array(apple_data + kiwi_data)
labels = np.array([0] * len(apple_data) + [1] * len(kiwi_data))

# Training and Test 90/10
d = zip(data, labels)
random.shuffle(d)
n = int(len(d) * 0.9)
X_train = np.array(map(lambda x: x[0], d[:n]))
y_train = np.array(map(lambda x: x[1], d[:n]))
X_test = np.array(map(lambda x: x[0], d[n:]))
y_test = np.array(map(lambda x: x[1], d[n:]))

Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)


# Create the model for the neural network
batch_size = 25
hidden_units = 50
nb_classes = 2
model = Sequential()
model.add(LSTM(output_dim=hidden_units, init='uniform', inner_init='uniform',
               forget_bias_init='one', activation='tanh', inner_activation='sigmoid',
               return_sequences=True, input_shape=X_train.shape[1:]))
model.add(LSTM(25))
model.add(Dense(nb_classes, activation='softmax'))
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd)

# Train the model
print("Train...")
history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=25, validation_data=(X_test, Y_test))
loss = model.evaluate(X_test, Y_test, batch_size=batch_size)

# Predictions
print "Training results:"
for prediction, label in zip(map(lambda x: np.argmax(x), model.predict(X_test)),
                             map(lambda x: np.argmax(x), Y_test)):
    print "The model has predicted {} and the label is {}".format(LABELS[prediction],
                                                                  LABELS[label])

test_audios = []
test_audios.append(get_mfcc_from_file('audio/my-apple.wav'))
test_audios.append(get_mfcc_from_file('audio/my-apple-2.wav'))
test_audios.append(get_mfcc_from_file('audio/my-kiwi.wav'))
test_audios.append(get_mfcc_from_file('audio/my-kiwi-2.wav'))

labels = [0, 0, 1, 1]
predictions = map(lambda x: np.argmax(x), model.predict(np.array(test_audios)))

print "Predicting different voices: "
for prediction, label in zip(predictions, labels):
    print "The model has predicted {} and the label is {}".format(LABELS[prediction],
                                                                  LABELS[label])

