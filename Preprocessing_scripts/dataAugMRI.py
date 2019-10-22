import cv2
import SimpleITK as sitk
import random
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk
from PIL import Image


def resize_image(img, new_size):
    """
    Resample brain MRI image to specified spacing, size_out and spacing out.
    img: The MRI image to resample.
    new_size: The spacing of the image we want.
    Function adapted from CODE/scripts_py/resample_image.py
    """
    sz_in, sp_in = img.GetSize(), img.GetSpacing()
    or_in, dir_in = img.GetOrigin(), img.GetDirection()
    new_size = [int(x) for x in new_size]
    new_spacing = [old_sz*old_spc/new_sz for old_sz, old_spc, new_sz in
                   zip(sz_in, sp_in, new_size)]
    t = sitk.Transform(3, sitk.sitkScale)
    # TODO: IF NEEDED, ADD GAUSSIAN SMOOTHING
    out_sitk = sitk.Resample(img, new_size, t, sitk.sitkLinear,
                             or_in, new_spacing,
                             dir_in, 0.0, sitk.sitkFloat64)
    return out_sitk


def gaussianNoiseMRI(img, stdv):
    # path = "/home/aitor/Escritorio/testingDA.nii.gz"
    # img = sitk.ReadImage(img_path)
    # img_array = sitk.GetArrayFromImage(img)

    gaussian = np.random.normal(0, stdv, img.shape)
    img_noise = img + gaussian

    # plt.imshow(img_noise)
    # plt.show()

    # y = sitk.GetImageFromArray(img_noise)
    # y.CopyInformation(img)
    # sitk.WriteImage(y, "noiseMRI.nii.gz")

    return img_noise


def flipMRIhor(img_array):

    # Flip horizontal
    img_array = img_array[:, :, ::-1]
    # plt.imshow(img_array[:, 87, :], cmap='gray')
    # plt.show()

    return img_array

def flipMRIver(img_array):
    # Flip vertical
    img_array = img_array[::-1, :, :]
    # plt.imshow(img_array[:, 87, :], cmap='gray')
    # plt.show()

    return img_array


def flipMRIboth(img_array):
    # Flip vertical y horizontal
    img_array = img_array[::-1, :, ::-1]
    return img_array


def random_rotation(img_path):
    img = sitk.ReadImage(img_path)
    img_array = sitk.GetArrayFromImage(img)
    # rotated_1 = np.zeros(img_array.shape)
    # rotated_2 = np.zeros(img_array.shape)

    img_array = Image.fromarray(img_array[:, 87, :,])
    # rotated_1 = img_array.rotate(25)
    # rotated_1 = np.array(rotated_1)

    rotated_1 = img_array.rotate(-25)

    # for i in range(img_array.shape[2]):
    #     img_array[:, :, i] = Image.fromarray(img_array[:, :, i])
    #     rotated_1[:, :, i] = img_array[:, :, i].rotate(25)
    #     rotated_2[:, :, i] = img_array[:, :, i].rotate(-25)
    #
    #
    rotated_1 = np.array(rotated_1)
    # rotated_2 = np.array(rotated_2)
    plt.imshow(rotated_1, cmap='gray')
    plt.show()


def crop_and_resize(img, size):
    """
    Create a random cropping from an input image, for a fixed size.
    The crop will be cubic.
    img: the image, in numpy. 3 dimensional.
    size: the size of the crop.
    """
    # img = sitk.ReadImage(img_path)
    # img = sitk.GetArrayFromImage(img)
    # plt.imshow(img[:, 87, :], cmap='gray')
    # plt.show()

    # get new coordinates
    # x y z could not coincide with actual x y z, just saying
    x = np.random.randint(0, img.shape[0] - size[0])
    y = np.random.randint(0, img.shape[1] - size[0])
    z = np.random.randint(0, img.shape[2] - size[0])
    i1 = img[x:x + size[0], y:y + size[0], z:z + size[0]]

    y = sitk.GetImageFromArray(i1)
    # y.CopyInformation(img)
    new_img = resize_image(y, img.shape)
    new_img = sitk.GetArrayFromImage(new_img)
    # plt.imshow(new_img[:, 87, :], cmap='gray')
    # plt.show()

    return new_img

    # return i1
