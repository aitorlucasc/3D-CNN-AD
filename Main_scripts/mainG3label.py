import argparse
import os
import configparser
import time
import pandas as pd
import numpy as np
np.random.seed(1714)
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.callbacks import TerminateOnNaN
from keras.callbacks import LearningRateScheduler
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from cnn3dG3label import CNN3D
from adSequence import adSequence

out_dir = "/homedtic/alucas/3labelsCNN/"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

t0 = time.time()

# 1. Load image paths
my_filtered_csv = pd.read_csv("/homedtic/alucas/DataCSV/mergedMRI3label.csv")
paths = my_filtered_csv['MRI_BIDS']
labels = my_filtered_csv['labels']

labels = to_categorical(labels)

# 2. Split into train, test, validation
X_train, X_test, y_train, y_test = train_test_split(paths, labels, train_size=0.8,
                                                    test_size=0.2)

# print(len(X_train))
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test,
                                                  test_size=0.5)

# print(len(y_train))
# print(len(y_val))

# 3. Configure the adSequence

# Create sequences of train/test (no really need for validation here)
adSeq = adSequence(X_train, y_train, batch_size=4)
adSeq_test = adSequence(X_test, y_test, batch_size=4)
adSeq_val = adSequence(X_val, y_val, batch_size=4)

# Create model
model = CNN3D(input_shape=(145, 172, 145, 1))
opt = Adam(lr=0.0001)

# Compile model
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Create callbacks
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

# Model checkpoint to save the training results
checkpointer = ModelCheckpoint(
    filepath=out_dir + "model_trained.h5",
    verbose=0,
    save_best_only=True,
    save_weights_only=True)

# CSVLogger to save the training results in a csv file
csv_logger = CSVLogger(out_dir + 'csv_log.csv', separator=';')


def lr_scheduler(epoch, lr):
    if epoch == 15:
        return lr
    elif epoch == 25:
        return lr * .1
    elif epoch == 35:
        return lr * .1
    else:
        return lr


lrs = LearningRateScheduler(lr_scheduler)

# Callback to terminate on NaN loss (so terminate on error)
NanLoss = TerminateOnNaN()

callbacks = [checkpointer, csv_logger, NanLoss, lrs, es]


# Train model
model.fit_generator(adSeq,
                    steps_per_epoch=None,
                    epochs= 45,
                    shuffle=True,
                    callbacks=callbacks,
                    verbose=1,
                    validation_data=adSeq_val)

# Evaluate test set
acc = model.evaluate_generator(adSeq_test, steps=None, max_queue_size=2, workers=1, verbose=1)
print("Test set: " + str(acc))

# Model is saved due to callbacks
print('The end.')
t1 = time.time()
print('Time to compute the script: ', t1 - t0)
