import os
import sys
import glob
import argparse
import numpy as np
from PIL import Image
from scipy.misc import toimage
from matplotlib import pyplot
from keras.datasets import cifar10
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications import mobilenet
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers import Input

def train(X_train, Y_train, X_test, Y_test):

    #define custom input shape
    input_shape = (128, 128, 3)
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = AveragePooling2D(pool_size=(7,7),strides=(1,1))(x)
    # let's
    # x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)

    predictions = Dense(10, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    # train the model on the new data for a few epochs
    # model.fit_generator(generator(X_train, Y_train), steps_per_epoch = 1500, epochs = 2)
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=25, batch_size=32)

    # print(model.summary())

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    # for i, layer in enumerate(base_model.layers):
    #    print(i, layer.name)

    # print(model.summary())

    # we chose to train the first layer, i.e. we will freeze
    # the first 80 layers and unfreeze the rest:
    # for layer in model.layers[:80]:
    #    layer.trainable = False
    # for layer in model.layers[80:]:
    #    layer.trainable = True
    #
    # # we need to recompile the model for these modifications to take effect
    # # we use SGD with a low learning rate
    # from keras.optimizers import SGD
    # model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
    #
    # # we train our model again (this time fine-tuning the top layer
    # # alongside the top Dense layers
    # model.fit(X_train, Y_train[0:1000], validation_data=(X_test, Y_test[0:100]), epochs=25, batch_size=32)
    #
    # Final evaluation of the model
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    # load model
    # model = load_model('models/MobileNet_CIFAR10_1000.h5', custom_objects={
    #                'relu6': mobilenet.relu6,
    #                'DepthwiseConv2D': mobilenet.DepthwiseConv2D})

    test_results = model.predict([X_test[50:59]], verbose=0, steps=None)
    print(test_results)

    for i in range(0, 9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(toimage(X_test[50 + i]))
        pyplot.title("Strength: " + str(np.amax(test_results[i])) +
        "\n Category: " + str(np.argmax(test_results[i])))
    pyplot.show()


    model.save('models/MobileNet_CIFAR10_50000.h5')


def loadData():

    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # load data
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    # normalize inputs from 0-255 to 0.0-1.0
    # X_train = X_train.astype('float32')
    # X_test = X_test.astype('float32')
    # X_train = X_train / 255.0
    # X_test = X_test / 255.0
    X_train_scaled = np.zeros(len(X_train))
    X_test_scaled = np.zeros(len(X_test))

    # one hot encode outputs
    Y_train = np_utils.to_categorical(Y_train)
    Y_test = np_utils.to_categorical(Y_test)
    num_classes = Y_test.shape[1]

    return X_train, Y_train, X_test, Y_test

def scaleUp(X_train, X_test):

    X_train_scaled = np.array([np.asarray((Image.fromarray(X_train[0], 'RGB')).resize((128, 128), Image.ANTIALIAS), dtype = np.float32) / 255.0])
    X_test_scaled = np.array([np.asarray((Image.fromarray(X_test[0], 'RGB')).resize((128, 128), Image.ANTIALIAS), dtype = np.float32) / 255.0])
    for i in range(1,1000):
        print("scaling training image:" + str(i))
        img = Image.fromarray(X_train[i], 'RGB')
        resized = np.asarray(img.resize((128, 128), Image.ANTIALIAS), dtype = np.float32)
        X_train_scaled = np.concatenate((X_train_scaled, np.array([resized])), axis=0)

    for j in range(1,100):
        print("scaling testing image:" + str(j))
        img = Image.fromarray(X_test[j], 'RGB')
        resized = np.asarray(img.resize((128, 128), Image.ANTIALIAS), dtype = np.float32)
        X_test_scaled = np.concatenate((X_test_scaled, np.array([resized])), axis=0)

    return X_train_scaled, X_test_scaled

def main():
    X_train, Y_train, X_test, Y_test = loadData()
    X_train, X_test = scaleUp(X_train, X_test)
    train(X_train, Y_train, X_test, Y_test)

if __name__ == '__main__':
    main()
