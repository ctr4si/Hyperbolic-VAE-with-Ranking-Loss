# Hyperbolic-VAE-with-Ranking-Loss
Hyperbolic Variational Autoencoder with Ranking Loss



## Python source code

1. Prerequisites:


  * Pytorch >= 0.3.0
  * python 2.7
  * Numpy >= 1.11.1
  * matplotlib

2. This code supports these three basic commands:
```
--batch_size, --epoch, --dimension command available
```
Data : MNIST, and planning to apply on ImageNet and CIFAR


## Instruction

1. Run `HypD5` with batch_size, epoch, dimension you want.
```python HypVAE.py --batch_size __ --epoch __ --dim __```
2. After the training ends, this code creates the following reconstruction outputs named as 'reconstruction_d(dim)_(epoch no).png' in folder 'results_hyp'.

## Results
The shape of the latent space of HVAE shows significant difference when it is compared to that of usual VAE.
