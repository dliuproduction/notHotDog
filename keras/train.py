import os
import sys
import glob
import argparse
import numpy as np
from PIL import Image
from scipy.misc import toimage
from matplotlib import pyplot as plt
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

IM_WIDTH, IM_HEIGHT, IM_CHANNELS = 128, 128, 3
NB_EPOCHS = 3
BAT_SIZE = 32
FC_SIZE = 1024
NB_LAYERS_TO_FREEZE = 70 # Number of layers to freeze for MobileNet
NB_CLASSES = 256 # Number of classes for Oxford-IIIT Pets

def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
      return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
      for dr in dirs:
        cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt

def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
      layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet

    Args:
      base_model: keras model excluding top
      nb_classes: # of classes

    Returns:
      new keras model with last layer
    """
    x = base_model.output
    # add a global spatial average pooling layer
    x = GlobalAveragePooling2D()(x)
    # add a fully-connected layer
    x = Dense(FC_SIZE, activation='relu')(x)
    # and a logistic layer with nb_classes
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    return model

def setup_to_finetune(model):
    """Freeze the bottom NB_LAYERS_TO_FREEZE and retrain the remaining top layers.

    note: NB_LAYERS corresponds to the top 2 blocks in the MobileNet architecture

    Args:
      model: keras model
    """
    for layer in model.layers[:NB_LAYERS_TO_FREEZE]:
       layer.trainable = False
    for layer in model.layers[NB_LAYERS_TO_FREEZE:]:
       layer.trainable = True
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

def train(args):
    """Use transfer learning and fine-tuning to train a network on a new dataset"""
    nb_train_samples = get_nb_files(args.train_dir)
    nb_classes = len(glob.glob(args.train_dir + "/*"))
    nb_val_samples = get_nb_files(args.val_dir)
    nb_epoch = int(args.nb_epoch)
    batch_size = int(args.batch_size)

    # data prep
    train_datagen =  ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
    )

    validation_generator = test_datagen.flow_from_directory(
        args.val_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size,
    )

    # create base model with custom input shape, pre-trained weights on imagenet, excluding top layer
    base_model = MobileNet(input_shape = (IM_WIDTH, IM_HEIGHT, IM_CHANNELS), weights="imagenet", include_top=False)

    # add custom top layers
    model = add_new_last_layer(base_model, nb_classes)

    # transfer learning
    setup_to_transfer_learn(model, base_model)

    history_tl = model.fit_generator(
        train_generator,
        nb_epoch=nb_epoch,
        samples_per_epoch=nb_train_samples,
        validation_data=validation_generator,
        nb_val_samples=nb_val_samples,
        class_weight='auto')

    # fine-tuning
    setup_to_finetune(model)

    history_ft = model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_val_samples,
        class_weight='auto')

    model.save(args.output_model_file)
    # model.load_weights(args.output_model_file)

    # import coremltools
    # coreml_model = coremltools.converters.keras.convert(model)
    # coreml_model.save('models/MobileNet_oxford_IIIT.mlmodel')

    if args.plot:
        plot_training(history_ft)

# def predict():
#     test_results = model.predict([X_test[50:59]], verbose=0, steps=None)
#     print(test_results)
#
#     for i in range(0, 9):
#         pyplot.subplot(330 + 1 + i)
#         pyplot.imshow(toimage(X_test[50 + i]))
#         pyplot.title("Strength: " + str(np.amax(test_results[i])) +
#         "\n Category: " + str(np.argmax(test_results[i])))
#     pyplot.show()

def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()


if __name__=="__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--train_dir")
    a.add_argument("--val_dir")
    a.add_argument("--nb_epoch", default=NB_EPOCHS)
    a.add_argument("--batch_size", default=BAT_SIZE)
    a.add_argument("--output_model_file")
    a.add_argument("--plot", action="store_true")

    args = a.parse_args()
    if args.train_dir is None or args.val_dir is None:
      a.print_help()
      sys.exit(1)

    if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
      print("directories do not exist")
      sys.exit(1)

    train(args)
