# Adversarial-Noise-Szegedy2013
This repo creates adversarial images following Szegedy et al. 2013.

The main script, ```create_adv_image.py```, reads images from the ```Original_Images``` folder, creates perturbed images, and stores them in the ```Perturbed_Images``` folder. The script also creates a summary plot comparing original and perturbed images and stores it in the ```Summary_Plots``` folder.

## Basic Usage
Run the following commands to create a perturbed version of the ImageNet validation image ```validation_0_label_91.jpg```.
- ```cd ./Code```
- ```python create_adv_image.py``` 

## Advanced Usage: process a custom image
Run the following commands to create a perturbed version of a custom image contained in a custom folder
- ```cd ./Code```
- ```python create_adv_image.py --img_name my_image.jpg --img_fld my_fld```

## Advanced Usage: process all images in the folder
Run the following commands to create perturbed versions of all the images in the ```Original_Images``` folder
- ```cd ./Code```
- ```python create_adv_image.py --img_name all```

## Advanced Usage: customize device and number of epochs
Run the following commands to change the number of training epochs and to perform the optimization on CPU
- ```cd ./Code```
- ```python create_adv_image.py --device cpu --n_epochs 1000``` 
