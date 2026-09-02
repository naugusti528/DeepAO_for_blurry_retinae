# DeepAO_for_blurry_retinae
An end-to-end deep learning project exploring blind image deconvolution for retinal imaging using a U-Net architecture enhanced with a model function tailored to retinal anatomical features. My original approach of a U-Net architecture failed, and on July 23rd 2026 I did a major overhaul of the architecture and started with a clean U-Net and a custom loss function.

This project investigates whether a convolutional neural network can reconstruct sharp retinal fundus images from synthetically blurred inputs. If it can, we can automate and improve retinal disease detection effectively. One particular condition that comes to mind is diabetic retinopathy.

## Background:
Retinal image of an eye is blurry due to the eye’s vision issues (nearsightedness, farsightedness, some other kind of myopia) or things like patients moving or a blurry camera. Poor image quality can reduce diagnostic accuracy and causes inconveniences. This project explores whether deep learning can automatically recover sharp retinal images from blurred observations.

## Objective:
To train a deep neural network capable of learning the inverse mapping from blurred retinal images to their corresponding sharp images. Instead of manually estimating the source of blur, this project formulates blind deconvolution as an image-to-image translation problem using supervised learning.

## Blind deconvolution:
We can do deconvolution while being blind to the cause of the blur.
I2I translation: mapping input images to desired output images (both are given)

## Model Architecture:
I use a U-Net deep learning architecture, along with a 2D Laplacian filter.
U-Net – encoder→bottleneck→decoder→[skip connections]
Type of learning: Supervised learning - direct learning
Loss functions: custom anatomical loss + Structural Similarity Index (SSIM) to mimic human vision

Update 7/15/26:
After training the model for 5 epochs, updating weights, and then 10 epoch training, small features such as blood vessels are still very much left out. The optic disc and fovea are recovered decently, but training the model even on 1000 epochs will not make the blood vessels any clearer.
The model architecture is U-Net with hybrid loss and encoder->bottleneck->decoder. As a further constraint, I'm imposing a pretrained VGG-16 network from Torchvision to target Perceptual/Feature Loss. This way, my model no longer ignores minute features and is still trained to deconvolve the more general, bigger features.

Update 9/2/2026:
I changed the training specs such that each batch was 4 images and each epoch had 250 batches, which means my model focuses on 1000 images at a time. After training for just 3 epochs, my model could recover blood vessels and other retinal features in crystal clear fashion. I stopped at 3 epochs due to the law of diminishing returns: the visual improvements I get from further training will exponentially decrease. There is a certain "sweet spot" where it deblurs almost completely, but just enough where further training will be better at wasting time instead of making nonnegligible improvements.

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
