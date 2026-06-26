# DeepAO_for_blurry_retinae
I use a U-Net architecture deep neural network to implement deconvolution on synthetically blurred retinal images.

## Background:
Retinal image of an eye is blurry due to the eye’s vision issues (nearsightedness, farsightedness, some other kind of myopia)

## Objective:
Take clean retinal images, artificially blur them by applying Gaussian noise, and then apply what’s called Deep-AO deblurring,
where we treat blind deconvolution as an image-to-image (I2I) translation task.

## Blind deconvolution:
We can do deconvolution while being blind to the cause of the blur.
I2I translation: mapping input images to desired output images (both are given)

## Architecture:
I use a U-Net deep learning architecture deep neural network to implement deconvolution on synthetically blurred retinal images.
U-Net – encoder→bottleneck→decoder→[skip connections]
Type of learning: Supervised learning - direct learning
Loss functions: MSE (standard) + Structural Similarity Index (SSIM) to mimic human vision

## Core Logic:
We have y = k*x + n, where x is the ground truth, y is the input blurred image, k is the point spread function, and n is noise. We are given y, and we must derive x.


## Data Specifications
The Kaggle dataset used for this project is 22 GB and is excluded via the .gitignore to maintain a lightweight and fast codebase.
The name of the dataset is "Eyepacs, Aptos, Messidor Diabetic Retinopathy" and it was created by Abdullah S. Canipek et al. under the username ascanipek.

## Usage
To run this project locally, use the following command to download the dataset onto your computer's local terminal:
kaggle datasets download -d ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy -p data/raw

This downloads the data as a zipfile. To unzip it, run:
unzip data/raw/eyepacs-aptos-messidor-diabetic-retinopathy.zip -d data/raw/extracted_images
