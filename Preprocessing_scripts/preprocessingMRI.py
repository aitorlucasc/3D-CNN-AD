import pandas as pd
import SimpleITK as sitk
import os
import numpy as np


def img3dNorm(img):
    img_sitk_name = img[-35:]
    # Read nii image
    img = sitk.ReadImage(img)

    # Convert it to array for better performance
    img_array = sitk.GetArrayFromImage(img)
    y = img_array
    #sizeImg = img_array.shape
    #max3d = np.zeros(sizeImg[2])

    #for i in range(sizeImg[2]):
    #    x = img_array[:, :, i]
    #    max3d[i] = np.max(x)

    #max_total = max(max3d)

    # Normalize to 0-1
    #y = img_array / max_total

    # Standardization
    y -= np.mean(y)
    y /= np.std(y)

    # Copy metadata from nii image
    y = sitk.GetImageFromArray(y)
    y.CopyInformation(img)

    # Output to file
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    sitk.WriteImage(y, out_dir + img_sitk_name)
    
    # Add the new path to the data csv
    row['MRI_BIDS'] = out_dir + img_sitk_name
    

# Read data from csv
my_filtered_csv = pd.read_csv("mergedMRIv3.csv")
out_dir = "/homedtic/alucas/sizeAndStd/"

# Iter through MRI_BIDS for image read and DX_bl for the y output
for i, row in my_filtered_csv.iterrows():
    img_path = row['MRI_BIDS']
    img3dNorm(img_path)

my_filtered_csv.to_csv('mergedMRIstd.csv', index=False)
