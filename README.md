# Exploring Generative Artificial Intelligence and Data Augmentation Techniques for Spectroscopy Analysis
This is repository is related to the information and methods discussed in our article. 

Check it out at:
https://pubs.acs.org/doi/10.1021/acs.chemrev.4c00815

# Installation

The software dependencies for this project are open-source and managed using [Miniconda](https://docs.anaconda.com/miniconda/) v24.1.2.
An environment YAML file is provided to create a custom environment and import the required dependencies.

Dependencies:
* Python v3.11.8
* NumPy v1.24.3
* Pandas v2.2.1
* Scikit-learn v1.5.2
* TensorFlow (with Keras) v2.14+ (new)
* PyTorch v2.2.1 (legacy notebooks)
* Jupyter notebook v7.0.8

The dependencies can be installed manually or using the environment.yml file provided. See the [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) for a quick guide on how to create a virtual environment.
Alternatively, [Google Colab](https://colab.google/) provides a free interactive Jupyter notebook environment with GPU access.

# Directories
This repository contains three folders located in the *src* directory that provide examples of methods described in our article. 
Each folder includes Jupyter notebooks that load the data from a CSV file named *cyclohexane.csv* in the *data* folder, located in the same directory as *src*.

The *Preprocessing* folder contains examples of pretreatment and preprocessing techniques corresponding to Section 3. Data Preprocessing.

The *Augmentation* folder contains examples of statistical and data augmentation methods, as described in Section 4. Chemometrics and Data Augmentation.

The *Generative AI* folder contains code that demonstrates how to train a Generative Adversarial Network and a Variational Autoencoder.
The VAE demonstrates how to apply raw versus preprocessed data, and the GAN demonstrates dense versus convolutional model architectures.

TensorFlow/Keras refactor:
- New Keras implementations of the GANs and VAE are available under `src/Generative AI/keras_models`.
- Files:
  - `gan_dense.py`: fully-connected GAN generator/discriminator
  - `gan_conv.py`: 1D convolutional GAN generator/discriminator
  - `vae.py`: encoder/decoder and a `VAE` model with custom training step

Quick start (Keras):
```python
from src.Generative AI.keras_models import (
    build_generator_dense, build_discriminator_dense,
    build_generator_conv1d, build_discriminator_conv1d,
    build_vae,
)

# Dense GAN for vectors of length 3276
G = build_generator_dense(output_dim=3276, noise_dim=128)
D = build_discriminator_dense(input_dim=3276)

# Compile a simple GAN training loop
from tensorflow import keras
gan = None  # use your preferred training loop, or DenseGAN in gan_dense.py

# Conv1D GAN for vectors of length 4101
G_conv = build_generator_conv1d(output_len=4101, noise_dim=128)
D_conv = build_discriminator_conv1d(input_len=4101)

# VAE for vectors of length N
vae = build_vae(input_dim=3276, latent_dim=20)
vae.compile(optimizer=keras.optimizers.Adam(1e-3))
# vae.fit(x_train, epochs=..., batch_size=...)
```

Note: Existing notebooks remain PyTorch-based and unchanged. You can import these new
Keras models from Python or adapt the notebooks to use them if preferred.

# Contact Us
For any questions related to this work, please contact the authors:
* A.flanagan18@universityofgalway.ie
* frank.glavin@universityofgalway.ie

# Acknowledgements
* This work was supported by the Taighde Éireann - Research Ireland Centre for Research Training in Artificial Intelligence (Grant No. 18/CRT/6223)
* This work was supported by the University of Galway
* The authors wish to express their sincere gratitude to Dr. Matthew Webberley and SK Pharmteco for providing the necessary facilities and supporting the data acquisition
