# Neuro-Style-Transfer

## Overview

This repository contains a simple implementation of image stylization using TensorFlow Hub's pre-trained model. The model takes two images as input - a content image and a style image - and produces a stylized image that combines the content of the first image with the style of the second image.

## Requirements

- Python 3.x
- TensorFlow 2.x
- TensorFlow Hub
- Matplotlib
- NumPy
- OpenCV

## Usage

1. Clone the repository using git clone.
2. Install the required dependencies using pip install -r requirements.txt.
3. Download the pre-trained model using the hub.load() function.
4. Load the content and style images using the load_image() function.
5. Run the stylization model using the model() function.
6. Display the stylized image using Matplotlib.

## Code Structure

- load_image(): Loads an image from a file path and preprocesses it for input to the model.
- model(): Loads the pre-trained stylization model and runs it on the input images.

## Example Use Case

- Load a content image (content.jpeg) and a style image (style.jpeg).
- Run the stylization model using the model() function.
- Display the stylized image using Matplotlib.

