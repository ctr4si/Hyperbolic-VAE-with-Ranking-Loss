# Hyperbolic-VAE-with-Ranking-Loss
This project is designed to analyse the effect of hyperbolic latent space on variational autoencoder. To design the new variational autoencoder, we added a modified ranking loss on VAE loss to strengthen the clustering property.



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


## Notes

 Since modified ranking loss requires positive pair and negative sets, and our initial dataset MNIST does not have such relationships between classes, we redefined the positive pair and negative sets for this experiment - Here, the positive pair set is a set of pairs of same class characters in each batch, and the negative set of a data x is all other characters that are in different class. 

## Instruction

1. Run `HypD5` with batch_size, epoch, dimension you want.
```python HypVAE.py --batch_size __ --epoch __ --dim __```
2. After the training ends, this code creates the following reconstruction outputs named as 'reconstruction_d(dim)_(epoch no).png' in folder 'results_hyp'.

## Results
The shape of the latent space of HVAE shows significant difference when it is compared to that of usual VAE.
