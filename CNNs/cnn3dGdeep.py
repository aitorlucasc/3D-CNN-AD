"""
CNN models.
File implementing the models to use for training and testing.
"""

from keras import models
from keras import layers
from keras import utils
from contextlib import redirect_stdout


def CNN3D(input_shape):
    """
    Define base 3D CNN implementation.
    Implement a 3D CNN for two-way classification following the architecture
    of Basu et al.
    """
    img_input = layers.Input(shape=input_shape)
    x = layers.Conv3D(11, (3, 3, 3),
                      name='conv1',
                      strides=(1, 1, 1), padding="valid")(img_input)
    #x = layers.BatchNormalization(axis=-1, name='bn1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), padding='valid', name='pool1')(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv3D(11, (3, 3, 3),
                      name='conv2', strides=(1, 1, 1), padding="valid")(x)
    #x = layers.BatchNormalization(axis=-1, name='bn2')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), padding='valid', name='pool2')(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv3D(11, (3, 3, 3),
                      name='conv3',
                      strides=(1, 1, 1), padding="valid")(x)
    #x = layers.BatchNormalization(axis=-1, name='bn3')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), padding='valid', name='pool3')(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv3D(11, (3, 3, 3),
                      name='conv4',
                      strides=(1, 1, 1), padding="valid")(x)
    #x = layers.BatchNormalization(axis=-1, name='bn4')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), padding='valid', name='pool4')(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv3D(11, (3, 3, 3),
                      name='conv5',
                      strides=(1, 1, 1), padding="valid")(x)
    #x = layers.BatchNormalization(axis=-1, name='bn4')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2), padding='valid', name='pool5')(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(4096, name='fc0', activation='relu')(x)
    x = layers.Dropout(0.2)(x)	

    x = layers.Dense(4096, name='fc1', activation='relu')(x)
    x = layers.Dropout(0.2)(x)
  

    x = layers.Dense(2, activation='softmax', name='fc3')(x)
    model = models.Model(img_input, x)

    with open('modelsummary.txt', 'w') as f:
        with redirect_stdout(f):
            utils.print_summary(model, line_length=110, print_fn=None)

    return model
