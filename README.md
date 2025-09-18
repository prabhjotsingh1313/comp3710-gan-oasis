WGAN for Brain Image Generation

This project implements a Wasserstein Generative Adversarial Network (WGAN) with Gradient Penalty (WGAN-GP) to generate realistic brain images using the OASIS dataset. The goal of this project is to generate synthetic brain images that mimic the real ones from the OASIS dataset.

Overview:

Generator (G): The generator network creates new brain images from random noise (latent vector).

Discriminator (D): The discriminator network tries to distinguish between real and fake images, guiding the generator to produce more realistic images over time.

WGAN with Gradient Penalty: We use the WGAN-GP approach to improve training stability by enforcing a 1-Lipschitz constraint on the discriminator.
