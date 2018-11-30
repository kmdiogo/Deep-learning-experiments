import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from pathlib import Path
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

number_of_classes = 10

# Reshape data to a single dimensional data input
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Limit input values from 0-1
x_train /= 255
x_test /= 255

y_train = np_utils.to_categorical(y_train, number_of_classes)
y_test = np_utils.to_categorical(y_test, number_of_classes)


DROPOUT_RATE = 0.1

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
#model.add(Dropout(DROPOUT_RATE))
#model.add(Dense(512))
#model.add(Activation('relu'))
#model.add(Dropout(DROPOUT_RATE))
model.add(Dense(10))
model.add(Activation('softmax'))

#model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(
    x_train,
    y_train,
    batch_size=128,
    nb_epoch=5,
    validation_data=(x_test, y_test),
    shuffle=True
)

# Save neural network structure
model_structure = model.to_json()
f = Path("fashion_mnist_structure.json")
f.write_text(model_structure)

# Save neural network's trained weights
model.save_weights("fashion_mnist_weights.h5")