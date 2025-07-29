# WGAN-GP vs Denoising Diffusion (DDPMs)
* ran on MNIST and CIFAR-10 datasets
* Both Models perform well on MNIST
* For CIFAR-10, WGAN-GP suffer from
  * blurred images
  * *artifacts*, and
  * hyper-parameter tuning
* DDPMs generate less blurred images and train faster
---

**Observations**    
*From WGAN-GP*
<p float="left">
<img src="CIFAR-10/images/sample_1_fake.png" style="width:150px; height:150px;" />
<img src="CIFAR-10/images/sample_2_fake.png" style="width:150px; height:150px;" />
<img src="CIFAR-10/images/sample_6_fake.png" style="width:150px; height:150px;" />
<img src="CIFAR-10/images/sample_7_fake.png" style="width:150px; height:150px;" /></p>

*From DDPM*
<p float="left">
<img src="Diffusion/Diffusers/CIFAR_10/images/sample_1.png" style="width:150px; height:150px;" />
<img src="Diffusion/Diffusers/CIFAR_10/images/sample_5.png" style="width:150px; height:150px;" />
<img src="Diffusion/Diffusers/CIFAR_10/images/sample_7.png" style="width:150px; height:150px;" />
<img src="Diffusion/Diffusers/CIFAR_10/images/sample_6.png" style="width:150px; height:150px;" /></p>

`All sample images can be viewed in the README files present in each folder.`
