import pandas as pd
import SimpleITK as sitk

# Read csv and configure the output path and your desired output image size
my_filtered_csv = pd.read_csv("mergedMRIv2.csv")
out_dir_img = "/homedtic/alucas/resizedG/"
new_size = [180, 192, 192]


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
                             dir_in, 0.0, sitk.sitkFloat32)
    return out_sitk


# Iterate over the dataframe
for i, row in my_filtered_csv.iterrows():
    # Get paths
    paths = row['MRI_BIDS']

    # Get the original image name
    img_sitk_name = paths[-35:]

    # Read the image, resize it and write it in the output path
    img_T1 = sitk.ReadImage(paths)
    # size = img_T1.GetSize()
    out_img = resize_image(img_T1, new_size)
    sitk.WriteImage(out_img, out_dir_img + img_sitk_name)
