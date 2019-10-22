"""
Data in-out.
FUnctions that deal with input and output of data and conversion to tensors.
Most of the data in/out funcionality is gathered from library DLTK:
https://github.com/DLTK
"""
import SimpleITK as sitk
import os
import numpy as np
from keras.utils import Sequence
import pandas as pd
import scipy.ndimage
import dataAugMRI as da


class adSequenceDA(Sequence):
    """
    Implement the Generator to train the model.
    Based on keras.utils.Sequence()
    """

    def __init__(self, x_set, y_set, batch_size):
        """
        Initialize class.
        x_set: list of paths to the images.
        y_set: associated classes.
        batch_size: Size of the training batch.
        norm: to normalize each image or not. Could be None, 'mean' or 'hist'
        norm_param: parameters for normalization
        color: if we adapt the image to a three channel tensor
        crop: crop at generator time
        x_set and y_set should have same length
        """
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        """Return length of the steps per epoch."""
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        """
        Return a full batch of data for training.
        This procedure must also do data augmentation steps.
        idx: internal parameter
        """
        'Generate one batch of data'
        # posar el for per llegir imatge a partir del path que entra

        # Get indices of each image
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_img = []
        for file in batch_x:
            # img = file
            img = load_img(file)

            # Data Augmentation
            if np.random.choice([0, 1], p=[0.5, 0.5]):
                img_noise = da.gaussianNoiseMRI(img, 0.3)
                img_noise = img_noise[:, :, :, np.newaxis]
                batch_img.append(img_noise)

            if np.random.choice([0, 1], p=[0.5, 0.5]):
                img_hor = da.flipMRIhor(img)
                img_hor = img_hor[:, :, :, np.newaxis]
                batch_img.append(img_hor)

            if np.random.choice([0, 1], p=[0.5, 0.5]):
                img_ver = da.flipMRIver(img)
                img_ver = img_ver[:, :, :, np.newaxis]
                batch_img.append(img_ver)

            if np.random.choice([0, 1], p=[0.5, 0.5]):
                img_both = da.flipMRIboth(img)
                img_both = img_both[:, :, :, np.newaxis]
                batch_img.append(img_both)

            if np.random.choice([0, 1], p=[0.5, 0.5]):
                img_crop = da.crop_and_resize(img, [95, 135, 95])
                img_crop = img_crop[:, :, :, np.newaxis]
                batch_img.append(img_crop)
	    
	    img = img[:, :, :, np.newaxis]
            batch_img.append(img)

        # concatenate arrays by 0 axis
        batch_img = np.stack(batch_img, axis=0)
        return batch_img, np.array(batch_y)


def load_img(path):
    """
    Load a single image from disk.
    path: Image path
    resample_spacing: spacing to resample img
    """
    sitk_t1 = sitk.ReadImage(path)
    img = sitk.GetArrayFromImage(sitk_t1)

    return img
