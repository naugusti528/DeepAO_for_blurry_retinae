# DeepAO_for_blurry_retinae
An end-to-end deep learning project exploring blind image deconvolution for retinal imaging using a U-Net architecture.

This project investigates whether a convolutional neural network can reconstruct sharp retinal fundus images from synthetically blurred inputs. If it can, we can automate and improve disease detection effectively.

This is a work in progress.

## Background:
Retinal image of an eye is blurry due to the eye’s vision issues (nearsightedness, farsightedness, some other kind of myopia) or things like patients moving or a blurry camera. Poor image quality can reduce diagnostic accuracy and may require patients to undergo repeat imaging. This project explores whether deep learning can automatically recover sharp retinal images from blurred observations.

## Objective:
To train a deep neural network capable of learning the inverse mapping from blurred retinal images to their corresponding sharp images. Instead of manually estimating the source of blur, this project formulates blind deconvolution as an image-to-image translation problem using supervised learning.

## Blind deconvolution:
We can do deconvolution while being blind to the cause of the blur.
I2I translation: mapping input images to desired output images (both are given)

## Model Architecture:
I use a U-Net deep learning architecture to implement deconvolution on synthetically blurred retinal images.
U-Net – encoder→bottleneck→decoder→[skip connections]
Type of learning: Supervised learning - direct learning
Loss functions: MSE (standard) + Structural Similarity Index (SSIM) to mimic human vision

## Core Logic:
We have y = k*x + n, where x is the ground truth (original image), y is the observed blurred image, k is the point spread function, and n is noise. We are given y, and we must derive x.

## Data Specifications
The Kaggle dataset used for this project is 22 GB and is excluded via the .gitignore to maintain a lightweight and fast codebase.
The name of the dataset is "Eyepacs, Aptos, Messidor Diabetic Retinopathy" and it was created by Abdullah S. Canipek et al. under the username ascanipek.

## Usage
To run this project locally, use the following command to download the dataset onto your computer's local terminal:
kaggle datasets download -d ascanipek/eyepacs-aptos-messidor-diabetic-retinopathy -p data/raw

This downloads the data as a zipfile. To unzip it, run:
unzip data/raw/eyepacs-aptos-messidor-diabetic-retinopathy.zip -d data/raw/extracted_images
