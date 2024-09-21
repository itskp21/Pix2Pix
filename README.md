## Overview
This repository contains the implementation of the Pix2Pix model, a conditional Generative Adversarial Network (cGAN) designed for image-to-image translation tasks. Pix2Pix takes in an input image and generates a corresponding output image based on the learned mapping. It can be applied to various tasks such as converting black and white images to color, generating maps from satellite images, and more.

## Features
Image-to-Image Translation: Supports a wide range of tasks including black & white to color conversion, sketch to image, and image style transfer.
Conditional GAN Framework: The generator learns to produce realistic images conditioned on an input image, while the discriminator helps improve the output by distinguishing between real and generated images.
Flexible: Train on custom datasets with different types of paired image-to-image tasks.

## Architecture
Pix2Pix uses a U-Net architecture for the generator and a PatchGAN for the discriminator.

Generator
U-Net Generator: The generator is based on an encoder-decoder architecture with skip connections between layers of the same size in the encoder and decoder. This allows the model to preserve fine details and structure of the input image.
Discriminator
PatchGAN Discriminator: Instead of classifying the entire image as real or fake, the PatchGAN classifies whether each N×N patch in the image is real or fake. This helps improve local coherence in the generated images.
Prerequisites

## Requirements
Python 3.x
CUDA (for GPU acceleration)
Python Libraries:
numpy
matplotlib
opencv-python
Pytorch
tqdm
