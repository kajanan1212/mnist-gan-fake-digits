# GAN that Generates Synthetic Digit Images

This repository contains the code for a programming assignment to implement a Generative Adversarial Network (GAN) that generates synthetic digit images resembling those from the MNIST dataset. The project involves building, training, and saving a GAN model, creating a fake dataset, evaluating the fake dataset with a CNN classifier, and performing analyses on the results.

## Project Structure
- `Generator G` - GAN's generator to create fake MNIST digits.
- `Discriminator D` - GAN's discriminator to differentiate between real and fake digits.
- `Classifier C` - CNN-based classifier to evaluate generated images.

## Steps

1. **Training the GAN**:
   - Load the MNIST dataset and use it to train the GAN.
   - Train until Generator G can produce indistinguishable fake digit images.
   - Save the generator and discriminator as `G.pkl` and `D.pkl` respectively.

2. **Generating the Fake Dataset**:
   - Generate 100 fake digit images using `Generator G`.
   - Save each image and the latent vector used to generate it in the `Fake_Digits` folder.
   - Ensure images are clearly identifiable as digits.

3. **Evaluating the Fake Dataset with a Classifier**:
   - Train `Classifier C` on the real MNIST dataset.
   - Save `Classifier C` as `C.pkl`.
   - Use the classifier to calculate classification errors on both the real dataset (S0) and the fake dataset (S1).

## Model Design
- **GAN Architecture**:
  - Generator: Takes a latent vector `z` (100 float values) as input and produces a synthetic image.
    ```log
    Generator(
      (linr1): Sequential(
        (0): Linear(in_features=100, out_features=256, bias=True)
        (1): ReLU()
        (2): Linear(in_features=256, out_features=784, bias=True)
        (3): Tanh()
      )
    )
    ```
  - Discriminator: Determines whether an image is real or fake.
    ```log
    Sequential(
      (0): Linear(in_features=784, out_features=256, bias=True)
      (1): LeakyReLU(negative_slope=0.2)
      (2): Linear(in_features=256, out_features=100, bias=True)
      (3): LeakyReLU(negative_slope=0.2)
      (4): Linear(in_features=100, out_features=1, bias=True)
      (5): Sigmoid()
    )
    ```
- **Classifier Architecture**:
  - CNN trained on the MNIST dataset to classify digit images.
    ```log
    CNN(
      (conv1): Sequential(
        (0): Conv2d(1, 36, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (1): ReLU()
        (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (conv2): Sequential(
        (0): Conv2d(36, 32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (1): ReLU()
        (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (out): Linear(in_features=1568, out_features=10, bias=True)
    )
    ```
