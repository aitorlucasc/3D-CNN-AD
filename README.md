# 3D-CNN-AD

*** Data and csv files are not provided ***

Repository that contains most of the files of my bachelor thesis called "3D Convolutional Neural Networks for Alzheimer's Disease"
    - Preprocessing_scripts: this folder contains the main preprocessing code.
        - preprocessingMRI.py: normalizes the MRI image values and stores the new ones.
        - resizeMRI.py: changes the default image size to another one.
        - dataAugMRI.py: implemented some data augmentation techniques as crops, flips, rotations and noise addition.
    - Sequence_scripts: as we have used 3D images, we can't load every image, so we have implemented a function that takes
                        small image batches and process them to our CNN. There is one script that implements data augmentation.
    - CNNs: the main CNN is cnn3dG.py, the other ones contain small variations.
    - Main_scripts: the main script is mainG.py, the others have small changes

Model architecture extrated from: "Analyzing Alzheimer’s Disease Progression from Sequential Magnetic Resonance Imaging Scans Using Deep 3D Convolutional Neural Networks".

Dataset comes from http://adni.loni.usc.edu/

The thesis can be downloaded in: http://hdl.handle.net/10230/42395

